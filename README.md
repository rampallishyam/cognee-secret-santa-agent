# Secret Santa Agent 🎅🎄

An intelligent Secret Santa matching system powered by Cognee knowledge graphs and LLM analysis. This application uses semantic understanding to create meaningful gift-giving pairs while respecting budget constraints and avoiding reciprocal matches.

## Features

### 🎲 Intelligent Matching
- **Cognee Hybrid Matching**: Combines graph database relationships with vector embeddings for intelligent pairing
- **Pure Vector Matching**: LanceDB-only semantic similarity matching for comparison
- **Strict Budget Enforcement**: Ensures participants are matched within compatible budget ranges
- **Reciprocity Avoidance**: Prevents A→B and B→A pairings (with fallback if necessary)

### 🤖 LLM-Powered Analysis
- **Match Quality Evaluation**: AI analyzes each pairing for compatibility
- **Compatibility Ratings**: HIGH/MEDIUM/LOW ratings with detailed explanations
- **Color-Coded Display**: 🟢 Green (HIGH), 🟡 Yellow (MEDIUM), 🔴 Red (LOW)
- **Dual Analysis**: Evaluates both Cognee and LanceDB matches

### 🎁 Gift Recommendations
- **Semantic Gift Search**: Uses Cognee to find personalized gift suggestions
- **Budget-Aware**: Recommends gifts within participant budgets
- **Interest-Based**: Matches gifts to participant interests and preferences

### 📊 Knowledge Graph Visualization
- **Interactive Graph**: Visualize relationships between participants
- **Browser Integration**: Open graph visualizations directly in browser
- **Semantic Connections**: See how interests and attributes connect participants

### 🎨 User Interface
- **Clean, Simple Design**: Minimal controls for ease of use
- **Michael Scott Mode**: Optional "rule breaker" participant for fun
- **Real-time Progress**: Visual progress indicators for long operations
- **Responsive Tables**: All participant interests visible with proper wrapping

## Prerequisites

- Python 3.11+
- OpenAI API key
- Virtual environment (venv)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/rampallishyam/cognee-secret-santa-agent.git
```

### 2. Set Up Environment

```bash
# Create virtual environment
uv venv .venv
# (if uv is not installed, install it using pip install uv)

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv sync
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.template .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini
```

### 4. Run the Application after minor setup

```bash
# build participant and gift databases (ETA: 15 min)
python src/secret_santa_agent/build_participant_database.py
python src/secret_santa_agent/build_gift_database.py

.venv/bin/python src/secret_santa_agent/gradio_app.py
```


The app will start at `http://127.0.0.1:7860`

## Usage Guide

### Loading Participants

1. Click **"📂 Load Demo Data"** to load 15 pre-configured participants
2. (Optional) Check **"Include Michael Scott"** to add a chaotic rule-breaker

### Generating Matches

#### Cognee Matches (Hybrid)
1. Navigate to **"🎲 Cognee Matches"** tab
2. Click **"🎲 Generate Matches"**
3. Wait for matching to complete
4. View results with match reasoning

#### Semantic Matches (Vector Only)
1. Navigate to **"🔍 Semantic Matches"** tab
2. Click **"🔍 Generate Semantic Matches (LanceDB Only)"**
3. Compare with Cognee results

### Analyzing Match Quality

1. Generate matches first (both types recommended)
2. Navigate to **"📊 Match Analysis"** tab
3. Click **"🤖 Analyze Match Quality with LLM"**
4. Review AI-generated compatibility ratings and explanations

### Finding Gift Ideas

1. Load participants first
2. Navigate to **"🎁 GiftScout"** tab
3. Select a participant from dropdown
4. Click **"🔍 Find Gifts"**
5. Browse personalized gift recommendations

### Visualizing Knowledge Graph

1. Navigate to **"🧠 Knowledge Graph"** tab
2. Click **"🔍 Visualize Knowledge Graph"**
3. Click **"🌐 Open in Browser"** to view interactive graph

## Project Structure

```
cognee-secret-santa-agent/
├── src/secret_santa_agent/
│   ├── gradio_app.py              # Main Gradio application
│   ├── agent.py                   # Core matching logic
│   ├── data/
│   │   ├── cognee_participants_db/    # Pre-built participant database
│   │   └── cognee_gifts_db/           # Pre-built gift database
│   └── helper_functions/          # Utility functions
├── .env.template                           # Environment configuration
└── pyproject.toml               # Python dependencies
```

## Key Technologies

- **[Cognee](https://github.com/topoteretes/cognee)**: Knowledge graph and vector database framework
- **[Gradio](https://gradio.app/)**: Web UI framework
- **[LanceDB](https://lancedb.com/)**: Vector database for semantic search
- **[OpenAI](https://openai.com/)**: LLM for match analysis and reasoning
- **Python 3.11**: Core programming language

## Configuration

### Matching Rules

The application uses the following hardcoded rules:

- **Interest Compatibility Weight**: 0.5 (moderate weighting)
- **Network Distance**: Minimum 2 hops between participants
- **Budget Matching**: Always strict (enforced)
- **Reciprocity Avoidance**: Always enabled (A→B means B≠A)

### LLM Model

Set your preferred model in `.env`:

```env
LLM_MODEL=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo, etc.
```

## Features in Detail

### Cognee Matching
Uses both graph relationships and vector similarity to find optimal pairings. The system:
1. Builds a knowledge graph from participant profiles
2. Queries Cognee for best matches based on interests and compatibility
3. Enforces budget constraints and reciprocity rules
4. Provides detailed reasoning for each match

### LanceDB Matching
Pure vector similarity approach:
1. Generates embeddings for each participant's profile
2. Performs semantic similarity search
3. Matches participants based on vector distance
4. Useful for comparison with Cognee's hybrid approach

### LLM Analysis
AI-powered evaluation of match quality:
- Analyzes giver and receiver profiles
- Considers interest overlap
- Evaluates budget compatibility
- Assigns compatibility rating (HIGH/MEDIUM/LOW)
- Provides human-readable explanations

## Troubleshooting

### App Won't Start
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv sync

# Check Python version
python --version  # Should be >=3.11 <=3.12
```

### No Matches Generated
- Verify OpenAI API key is set in `.env`
- Check that demo data is loaded
- Ensure budget constraints aren't too restrictive

### Graph Visualization Issues
- Ensure knowledge graph was built successfully
- Check browser compatibility (Chrome/Firefox recommended)
- Try regenerating the graph

## Development

### Building Custom Databases

**Participant Database:**
```bash
.venv/bin/python src/secret_santa_agent_simple/build_participant_database.py
```

**Gift Database:**
```bash
.venv/bin/python src/secret_santa_agent_simple/build_gift_database.py
```

### Adding New Features

The codebase is modular:
- **UI**: Modify `gradio_app.py`
- **Matching Logic**: Edit `agent.py`
- **Data**: Update CSV files and rebuild databases

## License

This project is for educational and demonstration purposes.

## Acknowledgments

- Built with [Cognee](https://github.com/topoteretes/cognee)
- UI powered by [Gradio](https://gradio.app/)
- Vector search via [LanceDB](https://lancedb.com/)
- LLM capabilities from [OpenAI](https://openai.com/)

---

**Made with ❤️ using Gradio • Secret Santa Agent v0.1.0**
