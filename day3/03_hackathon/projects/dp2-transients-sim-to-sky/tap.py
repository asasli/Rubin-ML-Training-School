"""RSP TAP access for DP2. The token is read from the RSP_TOKEN environment variable (set by ~/.rsp_env)
and is never printed or written anywhere. Row-level DP2 data may only be saved under DATA_DIR."""
import os, time
from pathlib import Path
import requests, pyvo

TAP_URL = "https://data.lsst.cloud/api/tap"
DATA_DIR = Path("/work/hdd/bcrv/asasli/later_data/dp2")


def data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(DATA_DIR, 0o700)
    return DATA_DIR


def service():
    tok = os.environ.get("RSP_TOKEN")
    if not tok:
        raise SystemExit("RSP_TOKEN is not set; run set_rsp_token.sh in a terminal first")
    s = requests.Session()
    s.headers["Authorization"] = "Bearer " + tok
    return pyvo.dal.TAPService(TAP_URL, session=s)


def query(adql, svc=None, label="", maxrec=None, retries=3):
    """Run an async TAP job and return a pandas DataFrame."""
    svc = svc or service()
    for attempt in range(retries):
        t0 = time.time()
        try:
            job = svc.submit_job(adql, maxrec=maxrec) if maxrec else svc.submit_job(adql)
            job.run(); job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=3600)
            job.raise_if_error()
            df = job.fetch_result().to_table().to_pandas()
            try:
                job.delete()
            except Exception:
                pass
            print(f"[tap] {label or adql[:60]}: {len(df):,} rows in {time.time()-t0:.0f} s", flush=True)
            return df
        except Exception as e:                       # the message never contains the token
            print(f"[tap] {label}: attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"TAP query failed: {label}")
