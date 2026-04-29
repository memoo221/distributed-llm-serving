**Title**  
Set up NGINX load-balancer simulation with 4 mock master nodes

**Summary**  
This PR adds a local Docker-based simulation for the distributed inference architecture so contributors can validate NGINX load balancing before the real master-node logic is implemented.

What is included:
- NGINX as the public entry point on port `8008`
- 4 simulated master nodes: `master1`, `master2`, `master3`, `master4`
- each mock master runs a lightweight FastAPI service on port `7000`
- a small traffic simulation script to verify request distribution across masters
- mock master routes refactored into `master/routers/mock_api.py`

Current request flow:
`Client -> NGINX:8008 -> Master Nodes:7000`

**Files Added / Updated**
- `docker-compose.yml`
- `nginx/nginx.conf`
- `master/Dockerfile.mock`
- `master/mock_api.py`
- `master/routers/mock_api.py`
- `tests/simulate_nginx_lb.py`
- `.dockerignore`

**How To Run**
1. Start Docker Desktop.
2. From the project root, run:
```powershell
docker compose up --build -d
```

3. Confirm all containers are up:
```powershell
docker compose ps -a
```

Expected:
- `distributed-nginx` is `Up`
- `master1`, `master2`, `master3`, `master4` are all `Up`

**How To Verify**
1. Check NGINX health:
```powershell
curl http://localhost:8008/nginx/health
```

Expected response:
```text
nginx ok
```

2. Check that requests are forwarded to a master:
```powershell
curl http://localhost:8008/
curl http://localhost:8008/health
curl -X POST http://localhost:8008/generate -H "Content-Type: application/json" -d "{\"prompt\":\"hello\",\"delay_ms\":500}"
```

Expected:
- HTTP `200`
- JSON response containing `master_id`

3. Verify load balancing:
```powershell
python tests/simulate_nginx_lb.py --requests 20 --concurrency 8 --delay-ms 1000
```

Expected:
- all requests return `status=200`
- the final summary shows traffic distributed across all 4 masters

Example expected distribution:
```text
master1: 5
master2: 5
master3: 5
master4: 5
```

**Failover Check**
To validate that NGINX reroutes traffic if one master is unavailable:
```powershell
docker compose stop master2
python tests/simulate_nginx_lb.py --requests 20 --concurrency 8 --delay-ms 1000
docker compose start master2
```

Expected:
- requests still succeed while `master2` is stopped
- `master2` disappears from the distribution summary during the stop window

**Important Ports**
- `8008`: NGINX load balancer
- `7000`: mock master node services

**Optional Local Backend Run**
If someone wants to run the separate backend app manually outside the Docker mock stack:
```powershell
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

This is separate from the NGINX simulation and should not be run on port `8008`.

**Notes**
- `nginx/nginx.conf` is mounted into the container, so config edits do not require rebuilding the image.
- Python code inside the mock master image does require rebuilds:
```powershell
docker compose up --build -d
```

**Current Scope**
This PR only validates the load balancer layer with simulated masters. Real master-node scheduling and worker-node logic will be implemented in later PRs.
