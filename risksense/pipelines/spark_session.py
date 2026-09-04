"""Shared SparkSession factory.

Local-first: defaults to ``local[*]`` so the full pipeline runs on a laptop
with zero cloud credentials (reproducibility constraint). On Databricks
Community Edition the active session is reused automatically.
"""

from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(app_name: str = "risksense") -> SparkSession:
    """Return the active SparkSession, creating a local one if needed.

    Parameters
    ----------
    app_name:
        Spark application name (visible in the Spark UI).
    """
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
