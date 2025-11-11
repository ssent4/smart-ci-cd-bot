# Smart CI/CD Bot

An intelligent log analysis system that predicts CI/CD pipeline failures and suggests automated fixes — built with **FastAPI**, **PyTorch**, **Docker**, and **Kubernetes**.

This project integrates machine learning into DevOps workflows, helping engineers proactively detect and resolve build issues.

---

## Features

* **ML-Powered Failure Detection:** Classifies CI/CD logs to identify likely failure causes
* **Context-Aware Suggestions:** Automatically recommends fixes based on error context
* **Dockerized Application:** Fully containerized with a production-ready FastAPI backend
* **Kubernetes Deployment:** Supports scalable multi-pod deployment via Minikube or ARO
* **CI/CD Pipeline Integration:** Ready for automation with GitHub Actions and Jenkins

---

## Architecture

```
FastAPI + PyTorch
   │
   ├── Docker Container
   │
   ├── Kubernetes Deployment (2 replicas)
   │
   └── Jenkins / GitHub Actions CI/CD Integration
```

---

## Tech Stack

| Component        | Technology                  |
| ---------------- | --------------------------- |
| Backend          | FastAPI                     |
| ML Framework     | PyTorch                     |
| Containerization | Docker                      |
| Orchestration    | Kubernetes (Minikube / ARO) |
| CI/CD            | Jenkins, GitHub Actions     |
| Data Handling    | Pandas, NumPy               |

---

## Local Development

### 1. Clone the repository

```
git clone https://github.com/<your-username>/smart-ci-cd-bot.git
cd smart-ci-cd-bot
```

### 2. Build the Docker image

```
docker build -t smart-cicd-bot:latest .
```

### 3. Run locally

```
docker run -p 8000:8000 smart-cicd-bot:latest
```

Then visit:
-> **[http://localhost:8000](http://localhost:8000)**

You should see:

```json
{"message": "Smart CI/CD Bot is running!"}
```

---

## Kubernetes Deployment (Minikube)

Start Minikube and build image inside its Docker env:

```
minikube start
eval $(minikube docker-env)
docker build -t smart-cicd-bot:latest .
```

Apply configs:

```
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
```

Check pods:

```
kubectl get pods
```

---

## Example API Call

### POST `/predict`

#### Request:

```json
{
  "message": "Task execution failed.",
  "pipeline_id": "CI-324",
  "stage_name": "Build",
  "job_name": "Compile",
  "task_name": "Compile code",
  "branch": "main",
  "user": "siva"
}
```

#### Response:

```json
{
  "prediction": 1,
  "probabilities": [0.02, 0.97, 0.01],
  "suggestion": "Test failure detected. Inspect test logs and recent changes."
}
```

---

## CI/CD Integration (Jenkins or GitHub Actions)

You can connect this repo to Jenkins or GitHub Actions to automatically:

* Rebuild and push Docker images
* Update Kubernetes deployments
* Run tests before merge or deployment

Example pipeline step:

```
kubectl rollout restart deployment smart-cicd-bot
```

---

## Future Enhancements

* Integrate with **Azure Red Hat OpenShift (ARO)** for enterprise-level scaling
* Add **log streaming and visualization dashboard**
* Implement **continuous retraining** for the ML model

---

## Author

**Siva Senthil Kumar**
*University of Illinois Urbana-Champaign*
B.S. Computer Science + Crop Sciences
[LinkedIn](https://linkedin.com/in/sivasenthilkumar) • [GitHub](https://github.com/<your-username>)

---
