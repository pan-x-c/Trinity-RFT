import sys
import unittest
from unittest import mock

from tests.tools import get_template_config
from trinity.cli import launcher


class TestLauncherMain(unittest.TestCase):
    def setUp(self):
        self._orig_argv = sys.argv.copy()

    def tearDown(self):
        sys.argv = self._orig_argv

    @mock.patch("trinity.cli.launcher.explore")
    @mock.patch("trinity.cli.launcher.train")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.bench")
    @mock.patch("trinity.cli.launcher.load_config")
    def test_main_run_command(self, mock_load, mock_bench, mock_both, mock_train, mock_explore):
        config = get_template_config()
        mapping = {
            "explore": mock_explore,
            "train": mock_train,
            "both": mock_both,
            "bench": mock_bench,
        }
        for mode in ["explore", "train", "both", "bench"]:
            config.mode = mode
            mock_load.return_value = config
            with mock.patch(
                "argparse.ArgumentParser.parse_args",
                return_value=mock.Mock(
                    command="run", config="dummy.yaml", dlc=False, plugin_dir=None
                ),
            ):
                launcher.main()
            mock_load.assert_called_once_with("dummy.yaml")
            mapping[mode].assert_called_once_with(config)
            mock_load.reset_mock()
            mapping[mode].reset_mock()

    @mock.patch("trinity.cli.launcher.stop_ray_cluster")
    @mock.patch("trinity.cli.launcher.setup_ray_cluster")
    @mock.patch("trinity.cli.launcher.both")
    @mock.patch("trinity.cli.launcher.load_config")
    @mock.patch("ray.init")
    def test_main_run_in_dlc(self, mock_init, mock_load, mock_both, mock_setup, mock_stop):
        config = get_template_config()
        namespace = f"{config.project}-{config.name}"
        config.mode = "both"
        config.log.level = "WARNING"
        config.log.group_by_node = True
        mock_setup.return_value = "auto"
        mock_load.return_value = config
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=mock.Mock(
                command="run", config="dummy.yaml", dlc=True, plugin_dir="/path/to/plugins"
            ),
        ):
            launcher.main()
        mock_init.assert_called_once()
        mock_init.assert_called_once_with(
            address="auto",
            namespace=config.ray_namespace,
            runtime_env={
                "env_vars": {
                    launcher.PLUGIN_DIRS_ENV_VAR: "/path/to/plugins",
                    launcher.LOG_DIR_ENV_VAR: config.log.save_dir,
                    launcher.LOG_LEVEL_ENV_VAR: config.log.level,
                    launcher.LOG_NODE_IP_ENV_VAR: "1",
                }
            },
        )
        mock_load.assert_called_once_with("dummy.yaml")
        mock_both.assert_called_once_with(config)
        mock_setup.assert_called_once_with(
            namespace=namespace,
        )
        mock_stop.assert_called_once_with(
            namespace=namespace,
        )

    @mock.patch("trinity.cli.launcher.studio")
    def test_main_studio_command(self, mock_studio):
        with mock.patch(
            "argparse.ArgumentParser.parse_args",
            return_value=mock.Mock(command="studio", port=9999),
        ):
            launcher.main()
        mock_studio.assert_called_once_with(9999)


if __name__ == "__main__":
    unittest.main()
