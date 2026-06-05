# Workflow CI Breast Cancer MLflow

Repository ini berisi MLflow Project dan GitHub Actions untuk retraining model secara otomatis.

## Cara Menjalankan Lokal

```bash
cd MLProject
mlflow run . --env-manager=local
```

Workflow GitHub Actions tersedia di `.github/workflows/main.yml` dan dapat dipicu lewat push ke `main` atau `workflow_dispatch`.
