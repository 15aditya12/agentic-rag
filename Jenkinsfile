pipeline {
    agent any

    stages {

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t agentic-rag:local .'
            }
        }

        stage('Load Image into Minikube') {
            steps {
                sh 'minikube image load agentic-rag:local'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh 'kubectl apply -f deployment.yaml'
                sh 'kubectl apply -f service.yaml'
            }
        }

        stage('Restart Deployment') {
            steps {
                sh 'kubectl rollout restart deployment agentic-rag'
            }
        }
    }
}
