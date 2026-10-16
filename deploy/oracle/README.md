# Deploying to Oracle Cloud Always Free (ARM)

Target: **VM.Standard.A1.Flex — 4 OCPU / 24 GB RAM = $0/month** under Oracle's
Always Free allocation (includes 200 GB boot volume + 10 TB/month egress).

Notes:

- Oracle's free tier is **ARM (ampere)**, so the app runs **natively** on the box
  (the repo's `Dockerfile` builds an amd64 image for other hosts and is not used here).
- Everything below the two console steps is automated over SSH by the deploy script
  `deploy/oracle/setup.sh` (venv, pinned deps, model download, systemd).

## 1. Create the instance (~5 min, needs a credit card for signup only)

1. Sign in at <https://cloud.oracle.com> → **Menu → Compute → Instances → Create instance**.
2. **Name**: `mmrag-demo`.
3. **Image**: change to **Canonical Ubuntu 24.04** (aarch64).
4. **Shape**: *Change shape* → `VM.Standard.A1.Flex` → **4 OCPU, 24 GB memory**
   (the full A1 flex allocation is Always Free eligible).
5. **SSH keys**: choose *Paste public keys* and paste:

   ```
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICz5H4hLC50m4DEZ4YyqmQNnONLvRqlddd4pmwBULmvo mmrag-oracle-deploy
   ```

6. **Boot volume**: keep the default size (needs ~15 GB total).
7. If you get **"Out of capacity for this shape"**: switch Availability Domain or
   region and retry (A1 capacity varies; AP-Mumbai/AP-Hyderabad/US-Ashburn usually work).
8. **Create instance** → wait for *Running* → copy the **Public IP**.

## 2. Open the demo port (~1 min)

**Virtual Cloud Networks** → the instance's VCN → **Security Lists** →
*Default Security List* → **Add Ingress Rules**:

| Field    | Value          |
|----------|----------------|
| Type     | Stateful       |
| Protocol | TCP            |
| Port     | `8501`         |
| Source   | `0.0.0.0/0` |

## 3. Hand over the public IP

The paired private key (`deploy/oracle/mmrag_key`, git-ignored) stays on the dev
machine. Provisioning then runs unattended:

1. source tree → `/home/opc/multi_rag`
2. `data/{index,pages,assets}` (~730 MB) streamed over SSH
3. `bash deploy/oracle/setup.sh` → Python 3.13 venv, pinned deps, models from
   Hugging Face, `mmrag.service`
4. live verification of `http://<ip>:8501` (UI + query + evidence overlays)

## On the box

- `mmrag.service` (systemd, auto-restart) runs `docker-entrypoint.sh`:
  FastAPI on `127.0.0.1:8000` (internal only) + Streamlit on `0.0.0.0:8501` (public).
- Logs: `journalctl -u mmrag -f`
- A weekly cron keep-alive burns ~2 min of CPU so Oracle doesn't reclaim the
  always-free instance for idleness (`# oracle-keepalive` in the script).
- Re-run `bash deploy/oracle/setup.sh` any time; it is idempotent.

## Cost

$0/month while it stays within Always Free limits. If Oracle retires the free
tier later, the same script works on any Ubuntu ARM VM (Hetzner/OVH cheap ARM
boxes, ~€4–6/mo).
