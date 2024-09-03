# Chainlit-RAG-Project

This repository contains a Chainlit application deployed to an AWS EC2 instance using Docker and GitHub Actions for continuous deployment (CD).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running Locally](#running-locally)
- [Deployment](#deployment)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Docker installed on your machine. [Download Docker](https://docs.docker.com/get-docker/)
- Python 3.x installed. [Download Python](https://www.python.org/downloads/)
- An AWS EC2 instance set up with Docker installed.
- GitHub account for cloning the repository and managing GitHub Actions.

## Setup Instructions

1. **Clone the Repository**

   Clone this repository to your local machine using:

   ```bash
   git clone https://github.com/your-username/my-chainlit-app.git
   cd my-chainlit-app
2. **Create a Virtual Environment**

   It's recommended to use a virtual environment for Python projects:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. **Install Dependencies**

   Install the Python dependencies listed in requirements.txt:
   ```bash
   pip install -r requirements.txt
4. **Create a .env File**

   In the root directory of the project, create a .env file and add your OpenAI API key:
   OPENAI_API_KEY=your_openai_api_key
   Replace your_openai_api_key with your actual OpenAI API key.

## Running Locally

To run the application locally, follow these steps:

1. **Build the Docker Image**

   Build the Docker image using the following command:

   ```bash
   docker build -t my-chainlit-app .
2. **Run the Docker Container**

   Run the Docker container with the required environment variables:

   ```bash
   docker run -d -p 8000:8000 --env-file .env my-chainlit-app
3. **Access the Application**

   Open your web browser and go to http://localhost:8000 to see your application running locally.

## Deployment

To deploy this application to an AWS EC2 instance using GitHub Actions and Docker, follow these steps:

1. **Set Up GitHub Secrets**

   In your GitHub repository, go to **Settings** > **Secrets and variables** > **Actions** > **New repository secret** and add the following secrets:

   - `EC2_SSH_KEY`: Your EC2 instance's private SSH key.
   - `DOCKER_USERNAME`: Your Docker Hub username.
   - `DOCKER_PASSWORD`: Your Docker Hub password.
   - `EC2_HOST`: Your EC2 instance's public IP address or DNS.
2. **Push Changes to main Branch**

   The GitHub Actions workflow will be triggered on every push to the main branch. Ensure your changes are pushed to the main branch:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin main
3. **GitHub Actions Workflow**

   The GitHub Actions workflow will automatically build and push a Docker image to Docker Hub, then deploy it to your AWS EC2 instance. You can monitor the progress in the Actions tab of    your GitHub repository.

