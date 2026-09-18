from google.cloud import bigquery

client = bigquery.Client(
    project="project-cd144a73-cc73-4010-bef"
)

rows = client.query(
    "SELECT 1 AS test"
).result()

for row in rows:
    print(row.test)