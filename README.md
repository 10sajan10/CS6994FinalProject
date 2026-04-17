# LRO PostgreSQL Docker Setup

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Docker Image file:https://drive.google.com/file/d/1bs_cHBNVIVP6eEhOQ7YoHixy8Ck_lfr2/view?usp=sharing

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

## Sharing the Database Image (437MB .tar)

Because the resulting `lro_postgres_image.tar` is larger than GitHub's 100MB file limit, it has been uploaded to Google Drive.

📥 **Download the Database Image:** [lro_postgres_image.tar (Google Drive)](https://drive.google.com/file/d/104fkVMOSoPA-O5F7ZxqdQ3L6YrLpBtR_/view?usp=sharing)

---

### Import (Receiver's End)
Once you download the(`.tar`) file from the link above to your local project directory, load it:
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

---

## The Data Pipeline (ETL)

This project follows a 3-step pipeline to ingest raw CSV data, normalize it into a relational database, and explore it.

### Step 1: Ingest Raw CSV Data
Run the python script to parse metadata headers from the LRO CSV files and batch-insert the data into the temporary `staging` table.
```bash
python ingest_csv.py
```
*(Make sure the CSV files are in the `Data/` folder).*

### Step 2: Normalize the Database
Execute the SQL script to move the flat staging data into structured tables (`site`, `variable`, `datastream`, etc.). First, copy the script into the container, then run it:
```bash
docker cp normalize_data.sql postgres:/normalize_data.sql
docker exec -it postgres psql -U admin -d database -f /normalize_data.sql
```

### Step 3: Explore the Data
Launch the Jupyter Notebook to verify the row counts and preview the normalized data.
```bash
jupyter notebook explore_staging.ipynb
```
*(Make sure to run `pip install pandas sqlalchemy psycopg2-binary notebook` first if you haven't).*

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
