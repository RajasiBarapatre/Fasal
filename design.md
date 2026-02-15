# Design Document: Fasal Neural Digital Twin

## Overview

The Fasal Neural Digital Twin is a cloud-native AI system built on AWS infrastructure that provides small and marginal farmers in India with economically-informed crop intervention recommendations. The system combines computer vision for disease detection, Transformer-based neural networks for crop simulation, and generative AI for multilingual explanations.

The core innovation is the Neural Digital Twin - a learned model that simulates the future state of a specific agricultural field under various intervention scenarios. By predicting yield outcomes and calculating ROI for each intervention, the system helps farmers avoid the "Silent Erosion" problem where economically unjustified interventions reduce profitability.

The architecture prioritizes affordability (serverless compute), accessibility (multilingual, smartphone-first), and reliability (low-connectivity tolerance) to serve the target demographic effectively.

## Architecture

### System Components

The system follows a microservices architecture with the following components:

1. **Frontend Layer (Streamlit)**
   - Smartphone-optimized web interface
   - Image capture and upload
   - Visualization of crop trajectories and ROI comparisons
   - Language selection and localized UI

2. **API Layer (AWS API Gateway + Lambda)**
   - RESTful API endpoints for all operations
   - Request routing and authentication
   - Rate limiting and throttling
   - Lambda functions for business logic orchestration

3. **AI/ML Layer**
   - **SageMaker CV Model**: Real-time endpoint for disease detection
   - **Neural Digital Twin**: Transformer-based model for crop simulation
   - **Amazon Bedrock**: Generative AI for multilingual explanations

4. **Storage Layer (Amazon S3)**
   - Crop images (Standard storage)
   - Synthetic training datasets (Standard-IA)
   - Historical field data (Standard-IA)
   - Model artifacts and checkpoints

5. **External Integrations**
   - Weather API for forecast data
   - Market price API for ROI calculations

### Data Flow

```
1. Image Upload Flow:
   Farmer → Streamlit → API Gateway → Lambda (Upload Handler) → S3

2. Disease Detection Flow:
   Lambda → SageMaker CV Endpoint → Lambda (Disease Processor) → DynamoDB

3. Simulation Flow:
   Lambda → Neural Digital Twin (Lambda or SageMaker) → ROI Calculator → Results Store

4. Explanation Flow:
   Results → Lambda → Bedrock (Claude/Titan) → Translated Explanation → Farmer

5. What-If Analysis Flow:
   Farmer Request → Lambda (Orchestrator) → Parallel Neural Twin Invocations →
   ROI Comparison → Bedrock Explanation → Ranked Results → Farmer
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Farmer (Mobile)                       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│                   (EC2 or App Runner)                        │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                              │
│              (Authentication, Rate Limiting)                 │
└─────┬──────────┬──────────┬──────────┬──────────────────────┘
      │          │          │          │
      ▼          ▼          ▼          ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
│ Upload  │ │ Detect  │ │Simulate │ │ Explain  │
│ Lambda  │ │ Lambda  │ │ Lambda  │ │  Lambda  │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬─────┘
     │           │           │           │
     ▼           ▼           ▼           ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│   S3    │ │SageMaker │ │  Neural  │ │ Bedrock  │
│ Storage │ │CV Model  │ │  Twin    │ │  API     │
└─────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Components and Interfaces

### 1. Image Upload Handler

**Responsibility**: Receive and validate crop images from farmers.

**Interface**:

```python
def upload_image(image_data: bytes, farmer_id: str, field_id: str, metadata: dict) -> UploadResult:
    """
    Upload a crop image for analysis.

    Args:
        image_data: Raw image bytes (JPEG/PNG)
        farmer_id: Unique identifier for the farmer
        field_id: Unique identifier for the field
        metadata: Additional context (GPS coordinates, timestamp, crop type)

    Returns:
        UploadResult containing S3 key and upload status
    """
```

**Implementation Details**:

- Validate image format (JPEG, PNG only)
- Check image size (max 10MB)
- Validate dimensions (min 640x640 pixels)
- Generate unique S3 key: `images/{farmer_id}/{field_id}/{timestamp}.jpg`
- Store metadata in DynamoDB for quick retrieval
- Return S3 key for downstream processing

### 2. Disease Detection Service

**Responsibility**: Analyze crop images and identify diseases with severity scores.

**Interface**:

```python
def detect_disease(s3_key: str) -> DiseaseDetectionResult:
    """
    Detect diseases in a crop image.

    Args:
        s3_key: S3 location of the image

    Returns:
        DiseaseDetectionResult containing list of detected diseases with severity scores
    """

class DiseaseDetectionResult:
    diseases: List[DetectedDisease]
    confidence: float
    processing_time_ms: int

class DetectedDisease:
    name: str
    severity_score: float  # 0-100
    confidence: float
    bounding_box: Optional[BoundingBox]
```

**Implementation Details**:

- Invoke SageMaker real-time endpoint with image
- Model architecture: ResNet50 or EfficientNet backbone with custom classification head
- Output: Multi-label classification with severity regression
- Post-processing: Filter detections below 70% confidence
- Severity scoring: Combine disease area coverage with intensity metrics
- Cache results in DynamoDB with TTL for repeated queries

### 3. Neural Digital Twin

**Responsibility**: Simulate future crop states and predict yield outcomes.

**Interface**:

```python
def simulate_trajectory(
    field_state: FieldState,
    intervention: Optional[Intervention],
    weather_forecast: WeatherForecast,
    simulation_days: int = 90
) -> CropTrajectory:
    """
    Simulate crop development over time.

    Args:
        field_state: Current state of the field (disease, growth stage, soil, etc.)
        intervention: Proposed intervention (None for baseline)
        weather_forecast: Weather predictions for simulation period
        simulation_days: Number of days to simulate

    Returns:
        CropTrajectory with daily predictions
    """

class FieldState:
    crop_type: str
    growth_stage: int  # Days since planting
    disease_severity: Dict[str, float]
    soil_moisture: float
    nutrient_levels: Dict[str, float]
    historical_yield: Optional[float]

class Intervention:
    type: str  # "pesticide", "fertilizer", "irrigation", etc.
    product_name: str
    application_rate: float
    cost_per_hectare: float
    application_day: int  # Day in simulation to apply

class CropTrajectory:
    daily_predictions: List[DailyPrediction]
    final_yield_kg_per_ha: float
    confidence_interval: Tuple[float, float]

class DailyPrediction:
    day: int
    disease_severity: Dict[str, float]
    biomass: float
    stress_level: float
    yield_potential: float
```

**Implementation Details**:

**Model Architecture**:

- Transformer encoder with temporal attention
- Input: Sequence of field states (current + historical)
- Positional encoding: Day of year + days since planting
- Weather embedding: Temperature, rainfall, humidity encoded as continuous features
- Intervention embedding: One-hot intervention type + continuous dosage
- Output: Sequence of future states (autoregressive generation)

**Training**:

- Trained on synthetic crop trajectory datasets
- Synthetic data generation: Crop growth models (DSSAT, APSIM) + noise
- Augmentation: Weather variations, intervention timing variations
- Loss function: MSE on yield + MSE on intermediate states + KL divergence for uncertainty

**Inference**:

- Autoregressive: Predict day t+1 from days 1..t
- Beam search for multiple trajectory samples
- Ensemble: Average predictions from 5 model checkpoints
- Uncertainty: Standard deviation across ensemble predictions

**Optimization for Lambda**:

- Model quantization (INT8) to reduce size
- ONNX runtime for faster inference
- Batch size 1 for single-field predictions
- Cold start mitigation: Provisioned concurrency for peak hours

### 4. ROI Calculator

**Responsibility**: Calculate return on investment for proposed interventions.

**Interface**:

```python
def calculate_roi(
    baseline_trajectory: CropTrajectory,
    intervention_trajectory: CropTrajectory,
    intervention: Intervention,
    market_price_per_kg: float
) -> ROIResult:
    """
    Calculate ROI for an intervention.

    Args:
        baseline_trajectory: Predicted trajectory without intervention
        intervention_trajectory: Predicted trajectory with intervention
        intervention: The intervention being evaluated
        market_price_per_kg: Current market price for the crop

    Returns:
        ROIResult with financial analysis
    """

class ROIResult:
    yield_increase_kg: float
    revenue_increase_inr: float
    intervention_cost_inr: float
    net_benefit_inr: float
    roi_percentage: float
    payback_days: int
    recommendation: str  # "RECOMMENDED", "NOT_RECOMMENDED", "MARGINAL"
```

**Implementation Details**:

- Yield difference: `intervention_trajectory.final_yield - baseline_trajectory.final_yield`
- Revenue increase: `yield_increase * market_price_per_kg`
- Net benefit: `revenue_increase - intervention_cost`
- ROI percentage: `(net_benefit / intervention_cost) * 100`
- Recommendation logic:
  - ROI > 15%: "RECOMMENDED"
  - ROI 0-15%: "MARGINAL"
  - ROI < 0%: "NOT_RECOMMENDED"
- Risk adjustment: Reduce expected yield by confidence interval lower bound
- Market price: Fetch from external API with fallback to historical average

### 5. What-If Analyzer

**Responsibility**: Compare multiple intervention scenarios in parallel.

**Interface**:

```python
def analyze_scenarios(
    field_state: FieldState,
    interventions: List[Intervention],
    weather_forecast: WeatherForecast
) -> ScenarioComparison:
    """
    Analyze multiple intervention scenarios.

    Args:
        field_state: Current field state
        interventions: List of interventions to compare (includes None for baseline)
        weather_forecast: Weather predictions

    Returns:
        ScenarioComparison with ranked results
    """

class ScenarioComparison:
    scenarios: List[ScenarioResult]
    best_scenario: ScenarioResult
    comparison_chart_data: dict

class ScenarioResult:
    intervention: Optional[Intervention]
    trajectory: CropTrajectory
    roi: ROIResult
    rank: int
```

**Implementation Details**:

- Parallel execution: Invoke Neural Digital Twin for each scenario concurrently
- Use Lambda async invocation or Step Functions for orchestration
- Always include baseline (no intervention) scenario
- Rank by ROI percentage (descending)
- Tie-breaking: Prefer lower upfront cost
- Generate comparison data for visualization:
  - Yield over time for each scenario
  - Cost vs. benefit scatter plot
  - ROI bar chart

### 6. Bedrock Explainer

**Responsibility**: Generate multilingual natural language explanations.

**Interface**:

```python
def generate_explanation(
    scenario_comparison: ScenarioComparison,
    language: str,
    farmer_context: FarmerContext
) -> Explanation:
    """
    Generate natural language explanation of recommendations.

    Args:
        scenario_comparison: Analysis results to explain
        language: Target language code (hi, ta, te, kn, mr, bn, gu, en)
        farmer_context: Farmer's background and preferences

    Returns:
        Explanation with recommendation text and implementation steps
    """

class FarmerContext:
    education_level: str
    farming_experience_years: int
    preferred_terminology: str  # "technical" or "simple"
    field_size_hectares: float

class Explanation:
    summary: str
    detailed_recommendation: str
    implementation_steps: List[str]
    expected_outcomes: str
    risks_and_considerations: str
    language: str
```

**Implementation Details**:

**Prompt Engineering**:

```
You are an agricultural advisor helping a farmer in India make decisions about crop interventions.

Context:
- Farmer's field: {field_size} hectares of {crop_type}
- Current situation: {disease_summary}
- Farmer's experience: {experience_years} years
- Language: {language}

Analysis Results:
- Baseline (no action): {baseline_yield} kg/ha, {baseline_revenue} INR
- Recommended intervention: {intervention_name}
- Expected yield with intervention: {intervention_yield} kg/ha
- Cost: {intervention_cost} INR
- Expected profit increase: {net_benefit} INR
- ROI: {roi_percentage}%

Instructions:
1. Explain the recommendation in simple, non-technical {language}
2. Use analogies familiar to farmers
3. Provide step-by-step implementation instructions
4. Mention expected timeline for results
5. Note any risks or precautions
6. Keep total explanation under 200 words

Generate the explanation:
```

**Bedrock Configuration**:

- Model: Claude 3 Haiku (cost-effective, multilingual)
- Temperature: 0.3 (consistent, factual)
- Max tokens: 500
- Fallback: If Bedrock fails, use template-based explanations

**Language Support**:

- Bedrock natively supports all target languages
- Post-processing: Validate output is in requested language
- Fallback: If language detection fails, regenerate with explicit language instruction

### 7. Weather Integration Service

**Responsibility**: Fetch and process weather forecast data.

**Interface**:

```python
def get_weather_forecast(
    latitude: float,
    longitude: float,
    days: int = 7
) -> WeatherForecast:
    """
    Fetch weather forecast for a location.

    Args:
        latitude: Field latitude
        longitude: Field longitude
        days: Number of days to forecast

    Returns:
        WeatherForecast with daily predictions
    """

class WeatherForecast:
    daily_forecasts: List[DailyWeather]
    source: str
    last_updated: datetime

class DailyWeather:
    date: date
    temperature_min_c: float
    temperature_max_c: float
    rainfall_mm: float
    humidity_percent: float
    wind_speed_kmh: float
```

**Implementation Details**:

- Primary source: OpenWeatherMap or India Meteorological Department API
- Caching: Cache forecasts for 6 hours in DynamoDB
- Fallback: If API unavailable, use historical seasonal averages
- Data validation: Check for unrealistic values, interpolate missing data
- Cost optimization: Batch requests for nearby fields

### 8. Synthetic Dataset Generator

**Responsibility**: Generate training data for Neural Digital Twin.

**Interface**:

```python
def generate_synthetic_dataset(
    crop_type: str,
    num_trajectories: int,
    region: str,
    season: str
) -> SyntheticDataset:
    """
    Generate synthetic crop trajectories for training.

    Args:
        crop_type: Type of crop (rice, wheat, cotton, etc.)
        num_trajectories: Number of trajectories to generate
        region: Geographic region for weather patterns
        season: Growing season

    Returns:
        SyntheticDataset with trajectories and metadata
    """

class SyntheticDataset:
    trajectories: List[SyntheticTrajectory]
    metadata: DatasetMetadata

class SyntheticTrajectory:
    field_state_sequence: List[FieldState]
    intervention: Optional[Intervention]
    weather_sequence: List[DailyWeather]
    final_yield: float
```

**Implementation Details**:

- Use crop growth models (DSSAT, APSIM) as simulators
- Parameter sampling: Sample soil types, planting dates, initial conditions
- Weather generation: Sample from historical weather distributions
- Intervention simulation: Random intervention types, timings, dosages
- Noise injection: Add realistic measurement noise
- Validation: Ensure generated trajectories are physically plausible
- Storage: Compress and store in S3 with Parquet format

## Data Models

### DynamoDB Tables

**1. Fields Table**

```python
{
    "PK": "FARMER#{farmer_id}",
    "SK": "FIELD#{field_id}",
    "crop_type": str,
    "field_size_hectares": float,
    "location": {
        "latitude": float,
        "longitude": float
    },
    "soil_type": str,
    "planting_date": str,  # ISO date
    "created_at": str,
    "updated_at": str
}
```

**2. Observations Table**

```python
{
    "PK": "FIELD#{field_id}",
    "SK": "OBS#{timestamp}",
    "image_s3_key": str,
    "disease_detection": {
        "diseases": List[dict],
        "severity_scores": dict
    },
    "growth_stage": int,
    "farmer_notes": str,
    "created_at": str
}
```

**3. Simulations Table**

```python
{
    "PK": "FIELD#{field_id}",
    "SK": "SIM#{simulation_id}",
    "baseline_trajectory": dict,
    "intervention_scenarios": List[dict],
    "roi_results": List[dict],
    "recommendation": str,
    "created_at": str,
    "ttl": int  # Auto-delete after 90 days
}
```

**4. Weather Cache Table**

```python
{
    "PK": "WEATHER#{lat}#{lon}",
    "SK": "FORECAST",
    "forecast_data": dict,
    "fetched_at": str,
    "ttl": int  # Auto-delete after 6 hours
}
```

### S3 Bucket Structure

```
fasal-neural-twin-{env}/
├── images/
│   └── {farmer_id}/
│       └── {field_id}/
│           └── {timestamp}.jpg
├── datasets/
│   └── synthetic/
│       └── {crop_type}/
│           └── {region}/
│               └── {season}/
│                   └── trajectories.parquet
├── models/
│   ├── cv-model/
│   │   └── model.tar.gz
│   └── neural-twin/
│       └── {version}/
│           └── model.onnx
└── results/
    └── {field_id}/
        └── {simulation_id}/
            └── visualization.json
```

## Error Handling

### Error Categories and Responses

**1. Input Validation Errors**

- Invalid image format → HTTP 400 with message "Supported formats: JPEG, PNG"
- Image too large → HTTP 413 with message "Maximum size: 10MB"
- Missing required fields → HTTP 400 with specific field name

**2. Service Unavailability Errors**

- SageMaker endpoint down → Retry 3 times with exponential backoff, then return HTTP 503
- Bedrock API error → Fall back to template-based explanations
- Weather API timeout → Use historical seasonal averages

**3. Model Prediction Errors**

- Low confidence detection → Return result with warning flag
- Simulation divergence → Return error with request to upload new image
- Unrealistic predictions → Cap predictions at historical maximum + 20%

**4. Storage Errors**

- S3 upload failure → Retry 3 times, then queue for later upload
- DynamoDB throttling → Implement exponential backoff with jitter
- Cache miss → Fetch from source and populate cache

**5. Connectivity Errors**

- Slow upload → Implement multipart upload with progress tracking
- Connection timeout → Queue request for retry when connectivity improves
- Partial response → Cache partial results and resume on reconnection

### Error Response Format

```python
class ErrorResponse:
    error_code: str
    message: str
    details: Optional[dict]
    retry_after_seconds: Optional[int]
    fallback_available: bool
```

### Graceful Degradation Strategy

1. **Disease Detection Failure**: Allow manual disease input by farmer
2. **Neural Twin Unavailable**: Provide rule-based recommendations
3. **Bedrock Unavailable**: Use pre-translated template explanations
4. **Weather API Down**: Use historical averages with warning
5. **ROI Calculation Error**: Show yield predictions without financial analysis

## Testing Strategy

### Overview

The testing strategy employs both unit tests and property-based tests to ensure correctness. Unit tests validate specific examples and edge cases, while property-based tests verify universal properties across randomized inputs. Each correctness property from this design will be implemented as a property-based test with minimum 100 iterations.

### Unit Testing

**Scope**:

- Individual function behavior with known inputs
- Edge cases (empty inputs, boundary values, null handling)
- Error conditions and exception handling
- Integration points between components
- Mock external dependencies (SageMaker, Bedrock, S3)

**Key Test Cases**:

1. Image upload with various formats and sizes
2. Disease detection with single/multiple diseases
3. ROI calculation with positive/negative/zero returns
4. Language selection and translation accuracy
5. Weather data parsing and fallback logic
6. S3 storage and retrieval operations
7. DynamoDB query patterns
8. API Gateway request/response handling

**Framework**: pytest for Python components

### Property-Based Testing

**Scope**:

- Universal properties that must hold for all valid inputs
- Randomized input generation to discover edge cases
- Invariants that should be preserved across operations
- Round-trip properties for serialization/deserialization

**Configuration**:

- Library: Hypothesis (Python)
- Minimum iterations: 100 per property
- Each test tagged with: `# Feature: fasal-neural-digital-twin, Property {N}: {description}`

**Test Data Generators**:

- Random field states with valid ranges
- Random interventions with realistic costs
- Random weather sequences
- Random crop trajectories
- Random disease severity scores (0-100)
- Random image metadata

## Correctness Properties

### What are Correctness Properties?

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees. Unlike unit tests that check specific examples, property-based tests verify that universal rules hold across many randomly generated inputs, helping discover edge cases and ensure comprehensive correctness.

### Property Reflection

After analyzing all acceptance criteria, I identified several opportunities to consolidate redundant properties:

- **ROI recommendation logic (3.3, 3.4)**: These can be combined into a single property that tests the complete recommendation logic based on ROI thresholds
- **Neural Twin weather integration (12.1, 12.2, 12.3)**: These all test that weather affects predictions and can be combined into one property about weather sensitivity
- **Orchestration flow (10.2, 10.3, 10.4)**: These test sequential pipeline steps and can be combined into one end-to-end orchestration property
- **Data organization (13.2, 13.4)**: Both test metadata properties of stored datasets and can be combined

### Properties

**Property 1: Disease severity scores are bounded**
_For any_ disease detection result, all severity scores must be between 0 and 100 inclusive.
**Validates: Requirements 1.2**

**Property 2: Multiple diseases have independent scores**
_For any_ image with multiple detected diseases, each disease must have its own severity score, and modifying one disease's severity should not affect others.
**Validates: Requirements 1.3**

**Property 3: Invalid images produce appropriate errors**
_For any_ image that fails quality checks (too small, corrupted, wrong format), the system must return an error message describing the specific quality issue.
**Validates: Requirements 1.5**

**Property 4: Trajectories have correct length**
_For any_ valid field state, the generated crop trajectory must contain exactly the requested number of daily predictions (default 90).
**Validates: Requirements 2.1**

**Property 5: Weather affects trajectory predictions**
_For any_ field state, generating trajectories with different weather forecasts must produce different yield predictions, demonstrating weather sensitivity.
**Validates: Requirements 2.2, 12.1, 12.2, 12.3**

**Property 6: Trajectories include yield predictions**
_For any_ generated crop trajectory, the final yield value must be present and non-negative.
**Validates: Requirements 2.3**

**Property 7: High disease severity triggers progression modeling**
_For any_ field state with disease severity > 20, the generated trajectory must show disease severity values that change over time (not constant).
**Validates: Requirements 2.4**

**Property 8: New observations update predictions**
_For any_ field with an existing trajectory, adding new observations and regenerating must produce a trajectory that differs from the original.
**Validates: Requirements 2.5**

**Property 9: ROI calculation uses all required inputs**
_For any_ intervention scenario, the ROI calculation must incorporate intervention cost, yield increase, and market price according to the formula: ROI = ((yield*increase * price - cost) / cost) \_ 100.
**Validates: Requirements 3.1, 3.2**

**Property 10: ROI-based recommendations follow threshold logic**
_For any_ intervention, if ROI > 15% the recommendation must be "RECOMMENDED", if ROI is between 0-15% it must be "MARGINAL", and if ROI < 0% it must be "NOT_RECOMMENDED".
**Validates: Requirements 3.3, 3.4**

**Property 11: ROI results include both formats**
_For any_ ROI calculation, the result must include both a percentage value and an absolute monetary value in INR.
**Validates: Requirements 3.5**

**Property 12: What-if analysis includes minimum scenarios**
_For any_ what-if analysis request, the results must include at least 3 different intervention scenarios.
**Validates: Requirements 4.1**

**Property 13: What-if analysis includes baseline**
_For any_ what-if analysis, one scenario must have intervention = None (baseline scenario).
**Validates: Requirements 4.2**

**Property 14: Scenarios are ranked by ROI**
_For any_ scenario comparison result, the scenarios must be sorted in descending order by ROI percentage.
**Validates: Requirements 4.3**

**Property 15: Tie-breaking prefers lower cost**
_For any_ two scenarios with ROI difference < 1%, the scenario with lower intervention cost must be ranked higher.
**Validates: Requirements 4.4**

**Property 16: Trajectories include weekly predictions**
_For any_ crop trajectory, yield predictions must be present at intervals of 7 days (±1 day tolerance for edge cases).
**Validates: Requirements 5.1**

**Property 17: Yield values have correct units**
_For any_ yield prediction, the value must be expressed in kilograms per hectare and be non-negative.
**Validates: Requirements 5.2**

**Property 18: Intervention visualization includes comparison**
_For any_ intervention scenario, the visualization data must include both the intervention trajectory and baseline trajectory for comparison.
**Validates: Requirements 5.3**

**Property 19: Predictions include confidence intervals**
_For any_ yield prediction, a confidence interval (lower bound, upper bound) must be present.
**Validates: Requirements 5.4**

**Property 20: Historical comparison when data exists**
_For any_ field with historical yield data, the prediction result must include a comparison field; for fields without historical data, the comparison field may be absent.
**Validates: Requirements 5.5**

**Property 21: Language selection is respected**
_For any_ supported language code (hi, ta, te, kn, mr, bn, gu, en), generating an explanation in that language must produce text in the requested language (verified by language detection).
**Validates: Requirements 6.2**

**Property 22: Explanations include implementation steps**
_For any_ generated explanation, the result must include a non-empty list of implementation steps.
**Validates: Requirements 6.4**

**Property 23: Simulation results are persisted**
_For any_ crop trajectory simulation, after generation the simulation must be retrievable from storage using its simulation ID.
**Validates: Requirements 7.4**

**Property 24: Storage operations retry on failure**
_For any_ storage operation that fails, the system must attempt the operation up to 3 times before returning an error.
**Validates: Requirements 7.5**

**Property 25: Failed uploads are queued**
_For any_ image upload that fails due to network error, the upload must be added to a retry queue.
**Validates: Requirements 8.1**

**Property 26: Response size is optimized**
_For any_ API response, the total payload size must be less than 500KB.
**Validates: Requirements 8.3**

**Property 27: Upload progress is tracked**
_For any_ image upload, progress callbacks must be invoked at least once during the upload process.
**Validates: Requirements 8.4**

**Property 28: Partial results are cached**
_For any_ operation that fails mid-execution, any completed partial results must be stored in cache for later retrieval.
**Validates: Requirements 8.5**

**Property 29: Pipeline orchestration flows correctly**
_For any_ image upload, the system must invoke components in sequence: disease detection → neural twin simulation → ROI calculation → explanation generation, with each step receiving output from the previous step.
**Validates: Requirements 10.2, 10.3, 10.4**

**Property 30: Component failures degrade gracefully**
_For any_ component failure (SageMaker, Bedrock, Weather API), the system must return a partial result with an error indicator rather than failing completely.
**Validates: Requirements 10.5**

**Property 31: Requests are batched when possible**
_For any_ set of concurrent requests for the same model, if batching is supported, the requests must be combined into a single batch invocation.
**Validates: Requirements 11.3**

**Property 32: Budget alerts are triggered**
_For any_ usage that exceeds configured budget thresholds, an alert must be sent to administrators.
**Validates: Requirements 11.5**

**Property 33: Weather fallback uses historical data**
_For any_ trajectory generation when weather API is unavailable, the system must use historical seasonal averages instead of failing.
**Validates: Requirements 12.4**

**Property 34: Weather changes trigger updates**
_For any_ field with an existing trajectory, if the weather forecast changes by more than 20% in any parameter, regenerating the trajectory must produce different predictions.
**Validates: Requirements 12.5**

**Property 35: Synthetic datasets are validated**
_For any_ generated synthetic dataset, all trajectories must pass validation checks (non-negative yields, valid date ranges, realistic parameter values) before storage.
**Validates: Requirements 13.1**

**Property 36: Datasets are organized and versioned**
_For any_ stored synthetic dataset, the S3 key must follow the pattern `datasets/synthetic/{crop_type}/{region}/{season}/{version}/` and include version metadata.
**Validates: Requirements 13.2, 13.4**

**Property 37: Dataset retrieval filters correctly**
_For any_ retraining request with specific criteria (crop type, region, season), only datasets matching all criteria must be retrieved.
**Validates: Requirements 13.3**

**Property 38: Large datasets are compressed**
_For any_ dataset exceeding 10GB in size, the stored version must be compressed (verified by file extension or metadata).
**Validates: Requirements 13.5**

**Property 39: Unauthenticated requests are rejected**
_For any_ API request without a valid authentication token, the system must return HTTP 401 Unauthorized.
**Validates: Requirements 14.3**

**Property 40: Farmers can only access own data**
_For any_ farmer attempting to access field data, the system must only return fields where the farmer_id matches the authenticated user's ID.
**Validates: Requirements 14.4**

**Property 41: Rate limiting is enforced**
_For any_ client sending more than the configured rate limit (e.g., 100 requests per minute), subsequent requests must be rejected with HTTP 429 Too Many Requests.
**Validates: Requirements 15.3**

**Property 42: High load triggers queuing**
_For any_ system state where active request count exceeds capacity, new non-urgent requests must be added to a queue rather than rejected.
**Validates: Requirements 15.5**

### Edge Cases and Examples

The following criteria are best tested as specific examples or edge cases rather than universal properties:

**Edge Case 1: No disease detected**
When an image contains no detectable disease, the system must return a severity score of exactly 0.
**Validates: Requirements 1.4**

**Example 1: All languages supported**
The system must successfully generate explanations for each of the 8 supported languages: Hindi, Tamil, Telugu, Kannada, Marathi, Bengali, Gujarati, and English.
**Validates: Requirements 6.1**

**Example 2: Training data retrieval**
The Neural Digital Twin must successfully retrieve synthetic datasets from S3 when training is initiated.
**Validates: Requirements 7.2**

**Example 3: Camera integration**
The smartphone interface must successfully invoke the device camera API when the user selects "Capture Image".
**Validates: Requirements 9.2**

**Example 4: SageMaker invocation**
When an image is uploaded, the system must invoke the SageMaker CV model endpoint via API Gateway.
**Validates: Requirements 10.1**

**Example 5: S3 encryption enabled**
All objects stored in S3 must have server-side encryption enabled (verified by checking object metadata).
**Validates: Requirements 14.2**
