# LRO PostgreSQL Docker Setup

## Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Git](https://git-scm.com/) installed

---

## 🚀 Quick Start (For Receivers/Reviewers)
Getting the database up and running with all 4 million rows of data is now completely automated! You do **not** need to manually run any Python scripts or import any database dumps.

### 1. Clone the Repository
```bash
git clone https://github.com/10sajan10/CS6694FinalProject.git
cd CS6694FinalProject
```

### 2. Build & Start the Database
```bash
docker compose up -d --build
```
*Note: The first time you start the container, it will take a minute or two because it is automatically executing the database creation scripts and unzipping the 4 million rows from the `init-scripts/` directory.*

### 3. Connect to the Database
| Parameter | Value            |
|-----------|------------------|
| Host      | `localhost`      |
| Port      | `5433`           |
| Database  | `database`       |
| User      | `admin`          |
| Password  | `password`       |

**Connect via psql using Docker:**
```bash
docker exec -it postgres psql -U admin -d database
```

**Connect via Python (psycopg2):**
```python
import psycopg2
conn = psycopg2.connect(
    host="localhost", port=5433,
    dbname="database", user="admin", password="password"
)
```

### 4. Stop the Container
```bash
docker compose down        # keeps data volume
docker compose down -v     # removes data volume too (WARNING: DELETES DATA)
```

---

## 🛠️ The Data Pipeline (For Developers)

If you are updating the raw data or modifying the database schemas, here is how the data was built from scratch.

### Step 1: Ingest Raw CSV Data
Run the python script to parse metadata headers from the LRO CSV files and batch-insert the data into the temporary `staging` table.
```bash
python ingest_csv.py
```
*(Make sure the raw CSV files are in the `Data/` folder).*

### Step 2: Normalize the Database
Execute the SQL script to move the flat staging data into structured tables (`site`, `variable`, `datastream`, etc.).
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

## 📂 Project Structure
```
Final Project/
├── Dockerfile                  # PostgreSQL 16 image definition
├── docker-compose.yml          # Service orchestration - Maps port 5433 to 5432
├── init-scripts/
│   ├── 01-create-schema.sql    # Auto-runs on first startup: Creates tables
│   └── 02-data-dump.sql.gz     # Auto-runs on first startup: Inserts 4M rows
├── ingest_csv.py               # Development: Ingests CSV to staging
├── normalize_data.sql          # Development: Moves staging to schema
└── explore_staging.ipynb       # Verifies counts and explores data
```
