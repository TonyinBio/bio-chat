from aws_cdk import (
    Stack,
    CfnOutput,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_secretsmanager  as secretsmanager,
    aws_iam as iam,
    aws_events as events,
    aws_events_targets as targets,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
)
from constructs import Construct


class CdkStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        neo4j_password = secretsmanager.Secret.from_secret_name_v2(
            self,
            "Neo4jPassword",
            "bio-chat/neo4j-password",
        )

        google_api_key = secretsmanager.Secret.from_secret_name_v2(
            self,
            "GoogleApiKey",
            "bio-chat/google-api-key",
        )

        service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "BioChatService",

            cpu=1024,
            memory_limit_mib=2048,
            desired_count=1,

            public_load_balancer=True,

            task_image_options=(
                ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                    image=ecs.ContainerImage.from_registry(
                        "250832562559.dkr.ecr.us-east-2.amazonaws.com/bio-chat:latest"
                    ),

                    container_port=8000,
                    environment={
                        "NEO4J_URI": "neo4j+s://ce4b1b9a.databases.neo4j.io",
                        "NEO4J_USERNAME": "ce4b1b9a",
                        "NEO4J_DATABASE": "ce4b1b9a",
                    },

                    secrets={
                        "NEO4J_PASSWORD": ecs.Secret.from_secrets_manager(neo4j_password),
                        "GOOGLE_API_KEY": ecs.Secret.from_secrets_manager(google_api_key),
                    }
                )
            ),
        )

        service.task_definition.execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AmazonECSTaskExecutionRolePolicy"
            )
        )
                
                # Email destination for alerts
        alerts = sns.Topic(
            self,
            "TaskFailureAlerts",
            display_name="BioChat task failures",
        )

        alerts.add_subscription(
            subscriptions.EmailSubscription("r.tony.sun@gmail.com")
        )

        # Detect crashes and failures to start for this ECS service
        failure_rule = events.Rule(
            self,
            "TaskFailureRule",
            event_pattern=events.EventPattern(
                source=["aws.ecs"],
                detail_type=["ECS Task State Change"],
                detail={
                    "clusterArn": [service.cluster.cluster_arn],
                    "group": [f"service:{service.service.service_name}"],
                    "lastStatus": ["STOPPED"],
                    "stopCode": [
                        "EssentialContainerExited",
                        "TaskFailedToStart",
                    ],
                },
            ),
        )

        failure_rule.add_target(targets.SnsTopic(alerts))

        CfnOutput(
            self,
            "LoadBalancerURL",
            value=service.load_balancer.load_balancer_dns_name,
        )