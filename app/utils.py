import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_data():
    """
    Load CI/CD logs dataset from Kaggle and return features & labels.
    """
    file_path = "ci_cd_logs.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "rahuljangir78/ai-driven-cicd-pipeline-logs-dataset",
        file_path
    )

    features = df[['message', 'pipeline_id', 'stage_name', 'job_name', 'task_name', 'branch', 'user']]
    df['status'] = df['status'].str.lower()

    def assign_label(row):
        msg = row['message'].lower()
        status = row['status'].lower()
        if "skipped" in msg:
            return 5
        elif "running" in msg:
            return 6
        elif status == "success":
            return 0
        elif status == "failed":
            if "test" in msg:
                return 1
            elif "deploy" in msg:
                return 2
            elif "dependency" in msg or "missing" in msg:
                return 3
            elif "env" in msg or "environment" in msg:
                return 4
        return 1

    labels = df.apply(assign_label, axis=1)

    valid_mask = labels.notna()
    features = features[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask].reset_index(drop=True)
    return features, labels

def get_fix(pred_idx, log_message=None):
    """
    Return a human-readable, actionable fix suggestion for a predicted failure type.
    Combines log keywords and model predictions.
    """

    if log_message:
        log_lower = log_message.lower()

        if "skipped" in log_lower:
            return "Task was skipped due to pipeline conditions. Check pipeline configuration."
        elif "running" in log_lower:
            return "Task is currently in progress. No action needed yet."
        elif "dependency" in log_lower or "missing" in log_lower:
            return "Missing dependency detected. Verify pipeline configuration and required packages."
        elif "env" in log_lower or "environment" in log_lower:
            return "Environment setup issue. Ensure consistent environment variables, OS, and libraries."
        elif "version" in log_lower:
            return "Version mismatch detected. Check build tool versions and dependencies."
        elif "test" in log_lower and "fail" in log_lower:
            return "Test failure detected. Review unit/integration tests and recent commits."
        elif "deploy" in log_lower and "fail" in log_lower:
            return "Deployment failed. Verify deployment scripts, target environment, and permissions."
        elif "timeout" in log_lower:
            return "Timeout occurred. Check network stability and service response times."
        elif "permission" in log_lower:
            return "Permission error. Verify user or service account access rights."
        elif "disk space" in log_lower or "storage" in log_lower:
            return "Insufficient disk/storage space. Free up space or increase limits."
        elif "memory" in log_lower or "out of memory" in log_lower:
            return "Memory issue detected. Check memory allocation and limits for the job."
        elif "exception" in log_lower or "error" in log_lower:
            return "An error or exception occurred. Review stack trace and logs for details."

    if pred_idx == 0:
        return "Pipeline succeeded. No action needed."
    elif pred_idx == 1:
        return "Test failure detected. Inspect test logs and recent changes."
    elif pred_idx == 2:
        return "Deployment failed. Check deployment scripts and target environment."
    elif pred_idx == 3:
        return "Dependency issue detected. Verify that all dependencies are installed."
    elif pred_idx == 4:
        return "Environment issue. Ensure proper configuration across stages."
    elif pred_idx == 5:
        return "Task was skipped due to pipeline conditions."
    elif pred_idx == 6:
        return "Task is currently in progress. No action needed yet."

    return "Pipeline may have failed. Check logs carefully."
