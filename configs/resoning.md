using YAML configuration files to separate parameters from code, which gives us several following advantages:

Environment flexibility - We can run the same code across development, staging, and production by just changing config files.
Experimentation efficiency - Data scientists can quickly test different model hyperparameters or feature sets without touching code.
Reproducibility - We can store exact configurations with model artifacts to ensure we can reproduce any result.
Non-technical accessibility - Business stakeholders can understand and sometimes modify key thresholds without needing to read code.
Deployment simplicity - Operations teams can adjust resource allocation or endpoints without requiring new code deployments.

YAML specifically is great because it's human-readable, supports comments for documentation, and handles nested structures well - perfect for the hierarchical nature of ML configurations.
This separation of configuration from implementation is a software engineering best practice that greatly improves maintainability and flexibility of ML systems in production