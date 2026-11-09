from prefect.schedules import Cron
from pipeline_prefect import flow_all


if __name__ == "__main__":
    flow_all.serve(
        name="ml-pipeline-all",
        schedule=Cron(
            "0 21 * * *",
            timezone="Africa/Tunis",
        ),
    )
