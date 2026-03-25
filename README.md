# LRO PostgreSQL Docker Setup

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

## Quick Start

### 1. Build & Start the Container
```bash
docker compose up -d --build
```

### 2. Connect to the Database
| Parameter | Value            |
|-----------|------------------|
| Host      | `localhost`      |
| Port      | `5432`           |
| Database  | `database`   |
| User      | `admin`      |
| Password  | `password`   |

**psql:**
```bash
docker exec -it postgres psql -U admin -d database
```

**Python (psycopg2):**
```python
import psycopg2
conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="database", user="admin", password="password"
)
```

### 3. Stop the Container
```bash
docker compose down        # keeps data volume
docker compose down -v     # removes data volume too
```

---

## Sharing the Image as a File

### Export (sender)
```bash
docker save -o lro_postgres_image.tar lro_postgres_image:latest
```
This creates `lro_postgres_image.tar` that you can share via USB, Google Drive, etc.

### Import (receiver)
```bash
docker load -i lro_postgres_image.tar
docker compose up -d
```

---

## Adding More Init Scripts
Drop additional `.sql` files into `init-scripts/`. They execute in alphabetical order on the **first** container startup only. To re-run them, remove the volume:
```bash
docker compose down -v
docker compose up -d --build
```

---

## Database Normalization (ETL)

After ingesting raw CSV data into the `staging` table using `ingest_csv.py`, you can normalize it into the structured schema:

1. **Run the Normalization Script**:
   ```bash
   docker exec postgres psql -U admin -d database -f /normalize_data.sql
   ```
2. **Verify the Results**:
   Open the `explore_staging.ipynb` notebook. The second section ("2. Normalized Data Verification") contains cells to count and preview the data in the structured tables (`site`, `variable`, `datastream`, etc.).

---

## Data Exploration (Jupyter Notebook)

To explore the ingested data:
1. Ensure your container is running: `docker compose up -d`
2. Install notebook dependencies:
   ```bash
   pip install pandas sqlalchemy psycopg2-binary notebook
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook explore_staging.ipynb
   ```
4. Run the cells to see the first 10 rows of the staging table.

---

## Project Structure
```
Final Project/
├── Dockerfile                  # PostgreSQL 16 image definition
├── docker-compose.yml          # Service orchestration
├── .dockerignore               # Build context exclusions
├── init-scripts/
│   └── 01-create-schema.sql    # Auto-runs on first startup
└── README.md
```

## Adding More Init Scripts
Drop additional `.sql` files into `init-scripts/`. They execute in alphabetical order on the **first** container startup only. To re-run them, remove the volume:
```bash
docker compose down -v
docker compose up -d --build
```
