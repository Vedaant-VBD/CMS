
# Cargo Management System

A FastAPI-based backend with a simple frontend to manage cargo on a space station. This project was developed for the NSH 2025 Hackathon.

## Features

- Container and Item Import: Upload CSV files for storage containers and items.
- Placement Optimization: Uses a Q-learning-based model to recommend item placements.
- Item Search & Retrieval: Search items by ID or name and log retrievals.
- Waste Identification: Detect expired or depleted items.
- Time Simulation: Simulate the passage of time to update item status.
- Logs: View historical actions like placements, retrievals, and waste flags.

## Tech Stack

- Backend: Python, FastAPI
- Frontend: HTML, CSS
- Containerization: Docker (based on Ubuntu 22.04)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/import/containers` | POST | Upload containers CSV |
| `/api/import/items` | POST | Upload items CSV |
| `/api/placement` | POST | Get placement suggestions |
| `/api/search` | GET | Search for an item |
| `/api/waste/identify` | GET | Identify waste items |
| `/api/simulate/day` | POST | Simulate passage of time |
| `/api/logs` | GET | Get logs of actions |

## Running with Docker

1. Clone the repository

```bash
git clone https://github.com/Vedaant-VBD/CMS.git
cd CMS
```

2. Build the Docker image

```bash
docker build -t cargo-system .
```

3. Run the Docker container

```bash
docker run -p 8000:8000 cargo-system
```

4. Open your browser to access the API

- http://localhost:8000
- http://localhost:8000/docs for Swagger UI

## Frontend

A simple HTML and CSS frontend is included to demonstrate basic functionality of the system.

## Repository Structure

```
├── main.py
├── train.py
├── evaluate.py
├── q_table.pkl
├── requirements.txt
├── Dockerfile
├── index.html
├── style.css
└── README.md
```

## Team

Developed as a part of the NSH 2025 Hackathon by [The Mutants (Vedaant Budakoti, Tamanna Mehra, Akshi Budhiraja, Nishant].
