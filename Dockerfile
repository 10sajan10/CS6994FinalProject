FROM postgres:16

# Set environment variables
ENV POSTGRES_USER=admin
ENV POSTGRES_PASSWORD=password
ENV POSTGRES_DB=database

# Copy initialization scripts (run in alphabetical order on first start)
COPY init-scripts/ /docker-entrypoint-initdb.d/

# Expose the default PostgreSQL port
EXPOSE 5432
