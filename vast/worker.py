"""
Vast.ai PyWorker configuration for Wan 2.2 I2V.

Vast.ai's infrastructure clones this via PYWORKER_REPO env var,
installs requirements.txt, and runs this file. The PyWorker proxy
forwards requests to our Flask handler_vast.py running on port 18000.
"""

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig

worker_config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=18000,
    model_log_file="/var/log/handler_vast.log",
    model_healthcheck_url="/health",
    handlers=[
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=600.0,
            workload_calculator=lambda _: 10000.0,
        )
    ],
    log_action_config=LogActionConfig(
        on_load=["Running on http://"],
        on_error=["RuntimeError", "CUDA out of memory", "Traceback"],
    ),
)

Worker(worker_config).run()
