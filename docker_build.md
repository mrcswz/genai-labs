## Build and push a docker image in Cloud Shell

docker build -t cymbal-docker-image -f Dockerfile .

docker image tag caf34e784294 us-central1-docker.pkg.dev/qwiklabs-gcp-02-a5948b2e0850/cymbal-artifact-repo/cymbal-docker-image:latest

docker push us-central1-docker.pkg.dev/qwiklabs-gcp-02-a5948b2e0850/cymbal-artifact-repo/cymbal-docker-image