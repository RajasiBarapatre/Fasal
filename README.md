# Fasal: AI Neural Digital Twin

An AI-powered Streamlit application that helps Indian farmers make economically optimal decisions about crop treatments using Neural Digital Twin technology.

## Features

- **Disease Detection**: Upload crop images for AI-powered disease severity assessment
- **Neural Digital Twin**: Simulates 90-day crop trajectories with and without treatment
- **ROI Calculator**: Determines if treatment is economically justified
- **Multilingual Support**: Explanations in English and Hindi using Amazon Bedrock
- **Interactive Visualizations**: Line charts showing yield predictions over time
- **Mobile-First Design**: Clean, agricultural-themed UI optimized for smartphones

## Architecture

The application integrates with AWS services:

- **Amazon SageMaker**: Computer vision model for disease detection (simulated)
- **Amazon Bedrock**: Claude 3 Haiku for multilingual explanations
- **AWS Lambda**: Serverless compute for processing (future integration)
- **Amazon S3**: Storage for images and datasets (future integration)

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure AWS credentials (for Bedrock integration):

```bash
aws configure
```

Make sure your AWS account has access to Amazon Bedrock in the us-east-1 region.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How It Works

1. **Upload Image**: Upload a crop image (JPG/PNG format)
2. **Set Parameters**: Adjust crop age, soil moisture, treatment cost, and market price
3. **Analyze**: Click "Analyze Field" to run the simulation
4. **Review Results**:
   - View disease severity score
   - See 90-day yield predictions (baseline vs. treatment)
   - Get ROI-based recommendation
   - Read AI-generated explanations in English and Hindi

## Simulation Logic

### Disease Detection

Simulates a SageMaker CV model that returns disease severity scores (0-100).

### Neural Digital Twin

Creates two 90-day trajectories:

- **Baseline**: No treatment, disease progresses naturally
- **Treatment**: With intervention, disease reduced by 70%

### ROI Calculation

```
Yield Increase = Treatment Yield - Baseline Yield
Revenue Increase = Yield Increase × Market Price
Net Benefit = Revenue Increase - Treatment Cost
ROI % = (Net Benefit / Treatment Cost) × 100

Recommendation:
- ROI > 15%: RECOMMENDED
- ROI ≤ 15%: NOT RECOMMENDED
```

## AWS Integration

### Amazon Bedrock Setup

The app uses Claude 3 Haiku for generating explanations. To enable:

1. Request access to Claude 3 Haiku in Amazon Bedrock console
2. Ensure your AWS credentials have `bedrock:InvokeModel` permissions
3. The app will automatically use Bedrock if available, otherwise falls back to templates

### Future Enhancements

- Real SageMaker endpoint integration for disease detection
- S3 storage for uploaded images
- DynamoDB for field history tracking
- Weather API integration
- Multi-crop support

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── design.md             # Detailed system design
└── requirements.md       # Functional requirements
```

## Mobile Optimization

The UI is optimized for smartphones with:

- Responsive layout
- Touch-friendly controls
- Progressive content loading
- Minimal bandwidth usage
- Green and white agricultural theme

## Cost Considerations

- Streamlit app: Free tier or minimal hosting costs
- Amazon Bedrock: ~$0.00025 per request (Claude 3 Haiku)
- Target: < ₹5 per recommendation

## License

This project is designed for educational and agricultural development purposes.

## Support

For issues or questions, please refer to the design and requirements documents included in this repository.
