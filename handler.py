import runpod


def handler(job):
    print("JOB RECEIVED:", job, flush=True)

    return {
        "success": True,
        "message": "SPOTC Serverless connection works",
        "job_id": job.get("id"),
        "input": job.get("input"),
    }


runpod.serverless.start({"handler": handler})
