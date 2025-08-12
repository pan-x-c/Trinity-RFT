import argparse
import io
import json
import uuid

import openai
from datasets import Dataset
from flask import Flask, jsonify, make_response, request

from .config_parser import ConfigParser
from .session import DataJuicerSession
from .utils import DataJuicerConfigModel

app = Flask(__name__)
openai_client = None  # Placeholder for OpenAI client, to be initialized later
sessions = {}


@app.route("/create", methods=["POST"])
def create():
    """Create a new data juicer session.

    Args:
        config (dict): Configuration parameters for the session.
        - Must include one of the following, and the priority is from high to low:
            - `operators` (`List[Dict]`): A list of operators with their configurations.
            - `config_path` (`str`): Path to the Data-Juicer configuration file.
            - `description` (`str`): The operator you want to use, described in natural language.

    Example:

    ```json
    {
        "config_path": "path/to/data_juicer_config.yaml",
        "operators": [
            {
                "operator1_name": {
                    "arg1": "value1",
                    "arg2": "value2"
                }
            },
            {
                "operator2_name": {
                    "arg1": "value1",
                    "arg2": "value2"
                }
            }
        ],
        "description": "Do somthing"
    }
    ```
    """
    config = request.json
    try:
        config = DataJuicerConfigModel.model_validate(config)
    except Exception as e:
        return jsonify({"error": f"Failed to parse config: {e}"}), 400

    session_id = str(uuid.uuid4())
    sessions[session_id] = DataJuicerSession(config, app.config["config_parser"])

    return jsonify({"session_id": session_id, "message": "Session created successfully."})


@app.route("/process", methods=["POST"])
def process():
    """
    Process uploaded experiences for a given session.
    Expects a multipart/form-data POST with a parquet file and session_id.
    Returns processed experiences and metrics as a parquet file and JSON.
    """
    session_id = request.json.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session ID not found."}), 404

    # Check for file in request
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]

    # Read parquet from uploaded file
    try:
        buffer = io.BytesIO(file.read())
        ds = Dataset.from_parquet(buffer)
    except Exception as e:
        return jsonify({"error": f"Failed to read parquet: {e}"}), 400

    # Process experiences using session
    session = sessions[session_id]
    try:
        # from_hf_datasets and to_hf_datasets should be imported from trinity.common.experience
        processed_ds, metrics = session.process(ds)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

    # Serialize processed experiences to parquet in-memory
    out_buffer = io.BytesIO()
    processed_ds.to_parquet(out_buffer)
    out_buffer.seek(0)

    # Return parquet file and metrics as response
    response = make_response(out_buffer.read())
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Content-Disposition"] = "attachment; filename=processed_experiences.parquet"
    # Add metrics as a custom header (JSON string)
    response.headers["X-Metrics"] = json.dumps(metrics)
    return response


@app.route("/close", methods=["POST"])
def close():
    """Close a data juicer session."""
    data = request.json
    session_id = data.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session ID not found."}), 404

    del sessions[session_id]
    return jsonify({"message": "Session closed successfully."})


def main(host="localhost", port=5005, debug=False):
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing Server")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=5005, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--openai-base-url",
        type=str,
        required=False,
        help="The OpenAI base url used by Data-Juicer Agent",
    )
    parser.add_argument(
        "--model_name", type=str, required=False, help="The model name used by Data-Juicer Agent"
    )
    args = parser.parse_args()
    openai_client = openai.OpenAI(
        base_url=args.openai_base_url,
        api_key=args.openai_api_key,
    )
    app.config["config_parser"] = ConfigParser(model_api=openai_client, model_name=args.model_name)
    main(host=args.host, port=args.port, debug=args.debug)
