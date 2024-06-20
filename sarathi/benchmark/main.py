import logging
import os

import yaml

from sarathi.benchmark.benchmark_runner import BenchmarkRunnerLauncher
from sarathi.benchmark.constants import LOGGER_FORMAT, LOGGER_TIME_FORMAT
from sarathi.benchmark.utils.random import set_seeds
from sarathi.config import BenchmarkConfig


def main() -> None:
    config = BenchmarkConfig.create_from_cli_args()

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f)

    set_seeds(config.seed)

    log_level = getattr(logging, config.log_level.upper())
    logging.basicConfig(
        format=LOGGER_FORMAT, level=log_level, datefmt=LOGGER_TIME_FORMAT
    )

    runner = BenchmarkRunnerLauncher(config)
    runner.run()


if __name__ == "__main__":
    main()