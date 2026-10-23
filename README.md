# Bio Chat
This project has two parts:
An LLM using retrieval augmented generation over a biomedical knowledge graph to answer 
questions. Served as a website.

An extractor that creates the knowledge graph by pulling and parsing scientific text
from PubMedCentral.

## Deploy to AWS with CDK

The CDK app in `cdk/` deploys `BioChatStack`: an ECS Fargate service behind a
public Application Load Balancer, with task failure alerts sent through SNS.
The stack currently targets **us-east-2** in the account selected by your AWS
credentials. It uses an existing ECR repository and Secrets Manager secrets;
it does not build the Docker image or deploy Neo4j.

### Prerequisites and configuration

Install Python 3, Node.js/npm, the AWS CLI, and Docker. Configure AWS credentials
with permission to bootstrap CDK and deploy the stack. For a named AWS profile,
set `AWS_PROFILE` before running the commands below.

From the repository root:

```bash
npm install -g aws-cdk
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION="$AWS_REGION"
export BIO_CHAT_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws sts get-caller-identity
python3 -m venv cdk/.venv
source cdk/.venv/bin/activate
python -m pip install -r cdk/requirements.txt
```

Before deploying, review `cdk/cdk/cdk_stack.py`:

- Set `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_DATABASE` for your database.
- Set `AURA_CLIENT_ID` and add `AURA_INSTANCEID` to the container `environment`
  mapping if using auto-resume. The current application reads `AURA_INSTANCEID`
  (without an underscore before `ID`). Set `AURA_AUTO_RESUME` to `false` if unused.
- Replace the SNS email subscription address with your alert destination.

In AWS Secrets Manager in **us-east-2**, create these secrets as plain-text
values, rather than JSON objects:

| Secret name | Value |
| --- | --- |
| `bio-chat/neo4j-password` | Neo4j database password |
| `bio-chat/google-api-key` | Google API key |
| `bio-chat/aura-client-sec` | Aura management API client secret |

The stack imports all three secrets, so they must exist even if auto-resume is
disabled, unless you also remove the unused secret reference from the stack.
The local `.env` file is not passed to the deployed ECS task.

### Build and publish the image

Create the ECR repository once, if it does not already exist:

```bash
aws ecr create-repository --repository-name bio-chat --region "$AWS_REGION"
```

From the repository root, log in to ECR, build for the stack's default x86 Fargate
architecture, and push the `latest` tag:

```bash
export BIO_CHAT_REGISTRY="$BIO_CHAT_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$BIO_CHAT_REGISTRY"
docker build --platform linux/amd64 -t "$BIO_CHAT_REGISTRY/bio-chat:latest" .
docker push "$BIO_CHAT_REGISTRY/bio-chat:latest"
```

See the [AWS ECR image publishing guide](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html).

### Deploy

Bootstrap the account and region once, then preview and deploy the stack:

```bash
cd cdk
cdk bootstrap "aws://$BIO_CHAT_ACCOUNT_ID/$AWS_REGION"
cdk synth BioChatStack
cdk diff BioChatStack
cdk deploy BioChatStack
```

Open `http://<LoadBalancerURL>` using the DNS name printed in the stack outputs.
Confirm the SNS subscription email to receive task failure alerts. The current
stack exposes HTTP; it does not configure a domain or TLS certificate.

For code updates, rebuild and push the image from the repository root. Because
the task definition uses the fixed `latest` tag, pushing an image alone does not
replace running tasks, and `cdk deploy` may report no changes. In the ECS console,
select the service created by `BioChatStack` and choose **Force new deployment**
to start tasks with the new image. Deploy CDK configuration changes separately
with `cdk deploy BioChatStack` from `cdk/`.

### Tear down

With the same AWS profile/account selected and the Python environment activated,
run from `cdk/`:

```bash
cdk destroy BioChatStack
```

Review and confirm the deletion prompt. This removes stack-managed resources,
including the ECS service, load balancer, and networking. It leaves the imported
`bio-chat` ECR repository and images, the three Secrets Manager secrets, the
external AuraDB instance, and the shared `CDKToolkit` bootstrap stack. Retained
CloudWatch log groups may also remain. Review these separately in AWS if you
want to remove them; deleting the application stack does not pause AuraDB.

See the [AWS CDK command reference](https://docs.aws.amazon.com/cdk/v2/guide/ref-cli-cmd.html)
for bootstrap, deployment, and deletion options.

## Local development

Set Neo4j password:
```console
export NEO4J_PASSWORD=...
```

Run the app:
```console
docker compose up -d
```

Enter docker container:
```
docker exec -it bio-chat-devcontainer-1 bash
```

Setup env without docker:
```console
conda create -n bio_chat python=3.11 -y
```
```console
conda activate bio_chat
```
```console
pip install -r requirements.txt
```
Also download [Ollama](https://ollama.com/download)

Process text and upload to neo4j:
```console
python ingest.py
```
Run a prompt:
```console
python starter.py
```

TODO: Citations
- ingest should add a new edge property
- retrieve should return edge property

TODO: Improve prompt?

TODO: Create NER evaluation dataset

TODO: Mess with CLIP/soft prompting?

TODO: Decide on predicates
- chemical
- source
- disposition 
- exposure_route 
- food
- health_effect
- organoleptic_effect 
- process
- role

Note to self:
plan:
- VectorIndex (VectorIndexRetriever) -> interfaces with embeddings in neo4j
- GraphIndex (KGRAGRetriever) -> interfaces with triplets and subgraph in neo4j 
- GRetriever -> combines the two
- will use both neo4jstores but have them point to the same db
