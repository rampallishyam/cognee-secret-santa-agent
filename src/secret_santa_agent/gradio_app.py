"""
Secret Santa Agent - Gradio Web Interface
==========================================
Interactive web application for intelligent Secret Santa matching.

Features:
---------
1. Participant Management
   - Load demo participants (15 included)
   - Michael Scott "rule breaker" mode
   - All interests displayed with word-wrap

2. Matching Approaches
   - Cognee Matches: Hybrid graph + vector semantic matching
   - Semantic Matches: Pure LanceDB vector similarity matching
   - Strict budget enforcement and reciprocity avoidance

3. LLM-Powered Analysis
   - AI evaluation of match quality using OpenAI
   - Color-coded compatibility ratings (HIGH/MEDIUM/LOW)
   - Detailed explanations for each pairing

4. Gift Recommendations
   - Semantic gift search using Cognee knowledge graphs
   - Budget-aware suggestions

5. Knowledge Graph Visualization
   - Interactive graph visualization
   - Browser-based viewing

Environment Variables Required:
- OPENAI_API_KEY: Your OpenAI API key
- LLM_PROVIDER: "openai" (default)
- LLM_MODEL: Model for analysis (e.g., "gpt-4.1-mini")

Usage:
    .venv/bin/python src/secret_santa_agent/gradio_app.py
    
Then open http://127.0.0.1:7860 in your browser.

Version: 0.1.0
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from debug_logger import logger, log_section, log_data
import gradio as gr

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")

# Import from the consolidated agent module in the same directory
try:
    from .agent import (
        SecretSantaAgent, 
        Participant, 
        Pairing, 
        MatchingRules, 
        BudgetLevel,
        CogneeSemanticMatcher,
        cognee
    )
except ImportError:
    # Fallback for running directly
    from agent import (
        SecretSantaAgent, 
        Participant, 
        Pairing, 
        MatchingRules, 
        BudgetLevel,
        CogneeSemanticMatcher, 
        cognee
    )

# Try importing search, but handle if it's not direct in cognee
try:
    from cognee.api.v1.search import search
except ImportError:
    search = None 
try:
    from cognee.api.v1.visualize.visualize import visualize_graph
except ImportError:
    visualize_graph = None


# Global state
agent = None
participants = []
past_pairings = []
graph_file_path = None
cognee_matches_cache = []      # Cache Cognee matches for analysis
lancedb_matches_cache = []     # Cache LanceDB matches for analysis
matches_cache = []  # Cache for statistics

def initialize_agent():
    """Initialize the Secret Santa Agent."""
    global agent
    load_dotenv()
    agent = SecretSantaAgent(llm_provider=os.getenv("LLM_PROVIDER", "openai"))
    return "Agent initialized successfully! 🎅"

async def load_demo_data(include_michael=False):
    """Load participants and past pairings from JSON files."""
    log_section("UI ACTION: LOAD DEMO DATA")
    log_data("Include Michael Scott", str(include_michael))
    global participants, past_pairings
    
    # Path adjustment for src/secret_santa_agent/ -> project root
    base_dir = Path(__file__).parent.parent.parent
    participants_file = base_dir / "data" / "participants_with_michael.json"
    pairings_file = base_dir / "data" / "past_pairings.json"
    
    participants = []
    past_pairings = []
    
    if participants_file.exists():
        with open(participants_file, 'r') as f:
            data = json.load(f)
            for item in data:
                # Filter out rule breakers if checkbox is unchecked
                if not include_michael and item.get('is_rule_breaker'):
                    continue
                    
                budget_level = BudgetLevel(item.get('budget_level', 'medium'))
                participants.append(Participant(
                    id=item['id'],
                    name=item['name'],
                    email=item['email'],
                    interests=item.get('interests', []),
                    budget=item.get('budget', 50),
                    budget_level=budget_level,
                    constraints=item.get('constraints', []),
                    preferences=item.get('preferences'),
                    is_rule_breaker=item.get('is_rule_breaker', False),
                ))
    
    if pairings_file.exists():
        with open(pairings_file, 'r') as f:
            data = json.load(f)
            past_pairings = [Pairing(**item) for item in data]
    
    # Count rule breakers
    rule_breakers = [p for p in participants if p.is_rule_breaker]
    
    # Init agent with data
    initialize_agent()
    agent.participants = participants
    agent.past_pairings = past_pairings
    
    # Get participant names for dropdown
    participant_names = [p.name for p in participants]
    
    # NOTE: Gift catalog is pre-built, no indexing needed at runtime
    # The pre-built database is at: src/secret_santa_agent/data/cognee_gifts_db/
    
    logger.info(f"Loaded {len(participants)} participants")
    return (
        f"✅ Loaded {len(participants)} participants ({len(rule_breakers)} rule breakers)",
        format_participants_display(),
        gr.update(choices=participant_names)
    )

def format_participants_display():
    """Format participants list as HTML table."""
    if not participants:
        return "<p>No participants loaded.</p>"
    
    html = "<table style='width: 100%; border-collapse: collapse;'>"
    html += "<tr style='background: #4a148c; color: white;'>"
    html += "<th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Name</th>"
    html += "<th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Interests</th>"
    html += "<th style='padding: 10px; text-align: left; border: 1px solid #ddd;'>Budget</th>"
    html += "</tr>"
    
    for p in participants:
        # Styling for rule breaker
        if p.is_rule_breaker:
            bg_color = "#e1b12c"   # Amber/gold for rule breakers
            text_color = "#2d3436"  # Dark text
            name_style = "font-weight: bold; color: #c0392b;"  # Red name
        else:
            bg_color = "#333"  # Dark background
            text_color = "#ecf0f1"  # Light text
            name_style = "font-weight: bold;"
        
        html += f"<tr style='background-color: {bg_color}; color: {text_color}; border-bottom: 1px solid #444;'>"
        html += f"<td style='padding: 8px; {name_style}'>{p.name} {('🚨' if p.is_rule_breaker else '')}</td>"
        # Show ALL interests with word wrapping - removed [:3] truncation and "..."
        html += f"<td style='padding: 8px; word-wrap: break-word; max-width: 400px;'>{', '.join(p.interests)}</td>"
        html += f"<td style='padding: 8px; text-align: center;'>${p.budget}</td>"
        html += "</tr>"
    
    html += "</table>"
    return html

async def generate_matches_async(progress=gr.Progress()):
    """Generate Secret Santa matches using Cognee with strict budget matching."""
    log_section("UI ACTION: GENERATE MATCHES")
    
    if not agent or not participants:
        return "❌ Please load demo data first!"
    
    # Hardcoded strict matching rules
    # - Budget matching: Always strict
    # - Reciprocity: Avoid A→B and B→A (allow as last resort only)
    # - Interest weight: Not used (Cognee handles semantic matching)
    # - Network distance: Not used
    rules = MatchingRules(
        interest_compatibility_weight=0.5,
        reciprocity_avoidance=True,
        min_network_distance=2,
        budget_matching="strict"  # Must be string: 'flexible', 'strict', or 'off'
    )
    
    agent.matching_rules = rules
    log_data("Matching Rules Updated", str(rules))
    
    # Ensure agent uses current UI participants state
    agent.participants = participants
    
    # Point to pre-built participant database
    try:
        # Save original data root to restore later
        original_data_root = cognee.config.data_root_directory
    except:
        import os
        original_data_root = os.path.join(os.path.expanduser("~"), ".cognee_system")

    # The helper exports to [db_path]/system_databases
    # We need to point to the parent of the 'databases' folder
    # Based on export structure: .../cognee_participants_db/system_databases/databases/..
    # Cognee usually expects the root that contains 'databases'.
    # Let's point to the system_databases folder where the export helper dumped the environment
    
    participant_db_path = Path(__file__).parent / "data" / "cognee_participants_db" / "system_databases"
    
    if participant_db_path.exists():
        cognee.config.data_root_directory = str(participant_db_path)
        logger.info(f"Switched Cognee to pre-built participant DB: {participant_db_path}")
        
        # We don't need to build knowledge graph, it's already built
        agent.knowledge_built = True
        agent.cognee_matcher.initialized = True
    else:
        logger.warning(f"Pre-built DB not found at {participant_db_path}, falling back to runtime build")
        progress(0.1, desc="Initializing Cognee Knowledge Graph...")
        await agent.build_knowledge_graph()

    logger.info("Cognee Knowledge Graph ready.")
    
    progress(0.3, desc="Finding optimal matches...")
    progress(0.3, desc="Finding optimal matches...")
    try:
        matches = await agent.generate_matches_with_cognee()
        
        # Restore original config for safety
        try:
            cognee.config.data_root_directory = original_data_root
        except:
            pass

        matches_cache = matches  # Store for statistics calculation
        logger.info(f"Generated {len(matches)} matches.")
    except Exception as e:
        logger.error(f"Error generating matches: {e}")
        return f"❌ Error generating matches: {str(e)}", None, None
    
    progress(0.8, desc="Formatting results...")
    
    # Cache matches for LLM analysis
    global cognee_matches_cache
    cognee_matches_cache = matches
    
    matches_html = format_matches_display(matches)
    
    return (
        f"✅ Successfully matched {len(matches)} pairs using Cognee!",
        matches_html
    )

def format_matches_display(matches):
    if not matches:
        return "<p>No matches yet.</p>"
    
    html = ""
    for m in matches:
        # Safe lookup with fallback
        giver = next((p for p in participants if p.id == m.giver_id), None)
        receiver = next((p for p in participants if p.id == m.receiver_id), None)
        
        # Handle missing participants gracefully
        if not giver or not receiver:
            giver_name = giver.name if giver else f"Unknown Participant ({m.giver_id})"
            receiver_name = receiver.name if receiver else f"Unknown Participant ({m.receiver_id})"
            bg_color = "#3a3a3a"  # Darker for error state
            border_color = "#ff6b6b"  # Red for warning
        else:
            giver_name = giver.name
            receiver_name = receiver.name
            bg_color = "#2a2a2a"
            border_color = "#667eea"
        
        text_color = "#e0e0e0"
              
        html += f"""
        <div style='background: {bg_color}; padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid {border_color}; color: {text_color};'>
            <h3 style='margin: 0 0 10px 0; color: {text_color};'>{giver_name} ➡️ {receiver_name}</h3>
            
            <div style='background: rgba(0,0,0,0.2); padding: 10px; border-radius: 6px; margin-top: 5px;'>
                <p style='margin: 0; color: #a29bfe; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px;'>🧠 Semantic Reasoning</p>
                <p style='margin: 5px 0 0 0; color: #dfe6e9; font-style: italic; line-height: 1.5;'>"{m.reasoning}"</p>
            </div>
        </div>
        """
    return html

def format_statistics(stats):
    if not stats: return "<p>No statistics available</p>"
    
    total_matches = stats.get('total_matches', 0)
    
    return f"""
    <div style='background: #1a1a1a; padding: 20px; border-radius: 10px;'>
        <h3 style='color: #667eea; margin-top: 0;'>📊 Event Statistics</h3>
        
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 15px;'>
            <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center; border: 2px solid #667eea;'>
                <div style='font-size: 2em; color: #a29bfe; margin-bottom: 5px;'>👥</div>
                <h2 style='color: #e0e0e0; margin: 5px 0;'>{stats['total_participants']}</h2>
                <p style='color: #999; margin: 0; font-size: 0.9em;'>Total Participants</p>
            </div>
            
            <div style='background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center; border: 2px solid #55efc4;'>
                <div style='font-size: 2em; color: #55efc4; margin-bottom: 5px;'>🎯</div>
                <h2 style='color: #e0e0e0; margin: 5px 0;'>{total_matches}</h2>
                <p style='color: #999; margin: 0; font-size: 0.9em;'>Matches Generated</p>
            </div>
        </div>
        
        <div style='background: #2a2a2a; padding: 12px; border-radius: 8px; border-left: 4px solid {'#55efc4' if stats.get('cognee_enabled') else '#ef5350'};'>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <span style='font-size: 1.5em;'>🧠</span>
                <div>
                    <strong style='color: #e0e0e0;'>Cognee Knowledge Graph:</strong>
                    <span style='color: {'#55efc4' if stats.get('cognee_enabled') else '#ef5350'}; margin-left: 8px;'>
                        {'✅ Active' if stats.get('cognee_enabled') else '❌ Inactive'}
                    </span>
                </div>
            </div>
        </div>
    </div>
    """

async def search_gift_ideas_cognee(participant_name, progress=gr.Progress()):
    """Cognee-powered gift suggestions using pre-built database."""
    log_section("UI ACTION: SEARCH GIFTS (COGNEE)")
    log_data("Target Participant", participant_name)

    if not participant_name:
        logger.warning("No participant name provided for gift search.")
        return "<p style='color: #ff6b6b;'>Please select a participant first.</p>"
    
    participant = next((p for p in participants if p.name == participant_name), None)
    if not participant:
        logger.error(f"Participant '{participant_name}' not found in loaded data.")
        return "<p style='color: #ff6b6b;'>Participant not found.</p>"
    
    progress(0.2, desc="Loading pre-built gift database...")
    
    
    # Build semantic query for gift search
    interests_str = ", ".join(participant.interests[:5])
    query = f"""List gift products for someone who likes {interests_str}.
Budget: ${participant.budget}
Preferences: {participant.preferences}

Return matching products with their names, prices, and brief descriptions. Use proper formatting."""
    
    try:
        # Temporarily switch Cognee to use the pre-built gift database
        from pathlib import Path
        import cognee
        from cognee.api.v1.search import SearchType
        
        # Save current data directory (property, not method)
        try:
            original_data_root = str(cognee.config.data_root_directory)
        except:
            # Fallback if property doesn't exist
            import os
            original_data_root = os.path.join(os.path.expanduser("~"), ".cognee_system")
        
        # Point to pre-built gift database v2 (proper structure with export helper)
        gift_db_path = Path(__file__).parent / "data" / "cognee_gifts_db" / "system_databases"
        cognee.config.data_root_directory = str(gift_db_path)
        
        progress(0.4, desc="Querying gift database...")
        logger.info(f"Switched Cognee to gift database: {gift_db_path}")
        
        # Query the pre-built gift database using CHUNKS for raw data retrieval
        # CHUNKS returns the actual indexed text segments, preserving exact prices and descriptions
        results = await cognee.search(
            query_text=query,
            query_type=SearchType.CHUNKS
        )
        
        # Restore original data directory
        cognee.config.data_root_directory = original_data_root
        logger.info(f"Restored Cognee to original directory")
        
        progress(0.8, desc="Formatting results...")
        
        # Format results as HTML with scrollable container
        html = f"""<div style='background: #1a1a1a; padding: 1.5rem; border-radius: 1rem; max-height: 600px; overflow-y: auto;'>
            <h3 style='color: #667eea; margin-top: 0;'>🎁 Cognee Gift Suggestions for {participant.name}</h3>
            <p style='color: #aaa;'><strong>Budget:</strong> ${participant.budget} | <strong>Interests:</strong> {interests_str}</p>
            <p style='color: #999; font-size: 0.9em;'><em>Using Pre-built Gift Database (65 curated products)</em></p>
            <hr style='border-color: #444;'>"""
        
        
        if not results:
            html += "<p style='color: #ff6b6b;'>No matching gifts found in database.</p>"
        else:
            display_count = min(len(results), 5)
            html += f"<p style='color: #74b9ff; margin: 10px 0;'>📊 Showing {display_count} matching product(s):</p>"
            
            import re
            
            gift_counter = 1
            for result in results[:5]:
                result_text = str(result).strip()
                
                # Clean up the raw result:
                # 1. Replace literal \n with actual newlines
                # 2. Remove trailing artifacts like }' or '}
                result_text = result_text.replace('\\n', '\n')
                result_text = re.sub(r"['\"}]+$", '', result_text)
                result_text = re.sub(r"^['\"{]+", '', result_text)
                
                # Extract fields using regex (now works with real newlines)
                name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', result_text)
                price_match = re.search(r'Price:\s*\$?(\d+)', result_text)
                category_match = re.search(r'Category:\s*(.+?)(?:\n|$)', result_text)
                desc_match = re.search(r'Description:\s*(.+?)(?:\n|$)', result_text)
                
                name = name_match.group(1).strip() if name_match else "Gift Product"
                price = f"${price_match.group(1)}" if price_match else "Price varies"
                category = category_match.group(1).strip() if category_match else ""
                description = desc_match.group(1).strip() if desc_match else ""
                
                html += f"""<div style='padding: 15px; margin: 12px 0; background: #2a2a2a; border-radius: 8px; border-left: 4px solid #667eea;'>
                    <h4 style='color: #74b9ff; margin: 0 0 8px 0;'>🎁 {name}</h4>
                    <p style='color: #81ecec; margin: 4px 0; font-size: 1.1em; font-weight: bold;'>{price}</p>
                    <p style='color: #dfe6e9; margin: 8px 0;'>{description}</p>
                    <p style='color: #636e72; font-size: 0.85em; margin: 4px 0;'>Category: {category}</p>
                </div>"""
                gift_counter += 1
        
        
        html += "</div>"
        logger.info(f"Generated {len(results)} gift suggestions for {participant.name}")
        return html
        
    except Exception as e:
        logger.error(f"Error searching gifts with Cognee: {e}")
        # Make sure to restore directory even if error occurs
        try:
            cognee.config.data_root_directory = original_data_root
        except:
            pass
        return f"<p style='color: #ff6b6b;'>Error: {str(e)}</p>"



def format_documentation_display():
    """Format documentation and QnA display."""
    return """
    <div style='background: #1a1a1a; padding: 2rem; border-radius: 1rem; color: #e0e0e0;'>
        <h2 style='color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;'>❓ QnA & Documentation</h2>
        <div style='margin-bottom: 2rem;'>
            <h3>👋 Welcome to the Secret Santa Agent!</h3>
            <p>This is not your average name-from-a-hat drawing. This application uses <strong>Cognee</strong> (a graph RAG framework) to understand the people involved and create meaningful connections.</p>
        </div>
    </div>
    """

async def visualize_knowledge_graph(progress=gr.Progress()):
    """Visualize Cognee knowledge graph."""
    global graph_file_path
    
    if not agent or not agent.knowledge_built:
        return "❌ Knowledge graph not built yet! Load data and generate matches first.", None, gr.update(visible=False)
    
    try:
        progress(0.5, desc="Generating knowledge graph visualization...")
        
        # Create file in accessible location
        output_dir = Path(__file__).parent.parent.parent / "artifacts"
        output_dir.mkdir(exist_ok=True)
        graph_file_path = str(output_dir / "knowledge_graph.html")
        
        if visualize_graph:
            await visualize_graph(destination_file_path=graph_file_path)
            return (
                f"✅ Knowledge graph saved! Click 'Open in Browser' to view.",
                graph_file_path,
                gr.update(visible=True)
            )
        else:
             return "❌ Cognee visualization module not found.", None, gr.update(visible=False)
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, gr.update(visible=False)

def open_graph_in_browser():
    """Open the knowledge graph in default browser."""
    global graph_file_path
    
    if not graph_file_path:
        return "❌ Generate the graph first!"
    
    try:
        import webbrowser
        import os
        
        # Convert to absolute path
        abs_path = os.path.abspath(graph_file_path)
        file_url = f"file://{abs_path}"
        
        # Open in browser
        webbrowser.open(file_url)
        
        return f"✅ Opened in browser: {abs_path}"
    except Exception as e:
        return f"❌ Error opening browser: {str(e)}"

async def generate_semantic_matches_lancedb(progress=gr.Progress()):
    """
    Generate matches using ONLY LanceDB vector search.
    No Cognee, no Kuzu graph - pure semantic similarity.
    """
    import lancedb
    import openai
    import re
    from pathlib import Path
    
    log_section("SEMANTIC MATCHING: LANCEDB ONLY")
    
    if not participants:
        logger.warning("No participants loaded for LanceDB matching")
        return "⚠️ Please load demo data first!", None
    
    progress(0.1, desc="Connecting to LanceDB...")
    
    # Connect to existing LanceDB (no Cognee imports)
    db_path = Path(__file__).parent / "data" / "cognee_participants_db" / "system_databases" / "cognee.lancedb"
    db = lancedb.connect(str(db_path))
    table = db.open_table("DocumentChunk_text")  # Full participant profiles
    
    logger.info(f"Connected to LanceDB at {db_path}")
    log_data("Table", "DocumentChunk_text")
    log_data("Total chunks", str(table.count_rows()))
    
    progress(0.3, desc="Computing semantic matches via vector search...")
    
    matches = []
    available_receivers = set(p.id for p in participants)
    
    for giver in participants:
        # Create query from giver's profile
        query_text = f"""Person interested in {', '.join(giver.interests[:5])}.
Budget: ${giver.budget}. Preferences: {giver.preferences or 'None'}"""
        
        log_data(f"Query for {giver.name}", query_text[:100])
        
        # Get embedding using OpenAI (same model as Cognee uses)
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=query_text
        )
        query_vector = response.data[0].embedding
        
        # Search LanceDB directly (vector similarity only)
        results = table.search(query_vector).limit(20).to_pandas()
        
        logger.info(f"Got {len(results)} vector search results for {giver.name}")
        
        # Find best available match
        matched = False
        for idx, row in results.iterrows():
            payload = row['payload']
            chunk_text = payload.get('text', '')
            
            # Extract participant ID from chunk text
            id_match = re.search(r'ID:\s*(\w+)', chunk_text)
            if not id_match:
                continue
            
            receiver_id = id_match.group(1)
            
            # Must be available and not self
            if receiver_id in available_receivers and receiver_id != giver.id:
                receiver = next((p for p in participants if p.id == receiver_id), None)
                
                if receiver:
                    distance = row.get('_distance', 0)  # Lower = more similar
                    
                    available_receivers.remove(receiver_id)
                    
                    matches.append(Pairing(
                        giver_id=giver.id,
                        receiver_id=receiver.id,
                        reasoning=f"Pure vector similarity (distance: {distance:.4f})"
                    ))
                    
                    log_data(f"MATCH: {giver.name} → {receiver.name}", f"Distance: {distance:.4f}")
                    matched = True
                    break
        
        if not matched:
            logger.warning(f"No match found for {giver.name} - using fallback")
            # Fallback: assign first available (excluding self)
            if available_receivers:
                # Get first available that's not the giver
                fallback_id = None
                for avail_id in available_receivers:
                    if avail_id != giver.id:
                        fallback_id = avail_id
                        break
                
                if fallback_id:
                    receiver = next(p for p in participants if p.id == fallback_id)
                    available_receivers.remove(fallback_id)
                    
                    matches.append(Pairing(
                        giver_id=giver.id,
                        receiver_id=receiver.id,
                        reasoning="Fallback match (no semantic match found)"
                    ))
                else:
                    # Only self remains - do a swap with last match to avoid self-assignment
                    logger.warning(f"Only self remaining for {giver.name}, performing swap with last match")
                    if matches:
                        # Take the last match and swap
                        last_match = matches[-1]
                        last_giver_id = last_match.giver_id
                        last_receiver_id = last_match.receiver_id
                        
                        # Current giver gets the last match's receiver
                        matches.append(Pairing(
                            giver_id=giver.id,
                            receiver_id=last_receiver_id,
                            reasoning="Swap match (avoiding self-assignment)"
                        ))
                        
                        # Update last match to give to current giver
                        matches[-2] = Pairing(
                            giver_id=last_giver_id,
                            receiver_id=giver.id,
                            reasoning=last_match.reasoning
                        )
                        
                        logger.info(f"Swapped: {giver.name} gets {last_receiver_id}, {last_giver_id} now gets {giver.id}")


    
    progress(0.8, desc="Formatting results...")
    
    # Cache matches for LLM analysis
    global lancedb_matches_cache
    lancedb_matches_cache = matches
    
    matches_html = format_matches_display(matches)
    
    logger.info(f"Generated {len(matches)} semantic matches using pure LanceDB")
    
    return f"✅ Generated {len(matches)} semantic matches using pure LanceDB vector search!", matches_html

async def analyze_matches_with_llm(progress=gr.Progress()):
    """
    Use OpenAI LLM to analyze match quality for both Cognee and LanceDB matches.
    Provides detailed reasoning on compatibility.
    """
    import openai
    
    log_section("LLM MATCH ANALYSIS")
    
    if not cognee_matches_cache and not lancedb_matches_cache:
        return "⚠️ Generate matches first (both Cognee and LanceDB)!"
    
    progress(0.1, desc="Preparing match analyses...")
    
    analyses = {
        'cognee': [],
        'lancedb': []
    }
    
    # Analyze Cognee matches
    if cognee_matches_cache:
        progress(0.2, desc=f"Analyzing {len(cognee_matches_cache)} Cognee matches...")
        
        for i, match in enumerate(cognee_matches_cache):
            giver = next((p for p in participants if p.id == match.giver_id), None)
            receiver = next((p for p in participants if p.id == match.receiver_id), None)
            
            if not giver or not receiver:
                continue
            
            # Create LLM prompt
            prompt = f"""Analyze this Secret Santa pairing and determine if it's a good match:

GIVER: {giver.name}
- Interests: {', '.join(giver.interests)}
- Budget: ${giver.budget}
- Preferences: {giver.preferences or 'None'}

RECEIVER: {receiver.name}
- Interests: {', '.join(receiver.interests)}
- Budget: ${receiver.budget}
- Preferences: {receiver.preferences or 'None'}

Matching Method: Cognee (Graph + Vector Database)
Cognee's Reasoning: {match.reasoning}

Provide a brief analysis (2-3 sentences):
1. Is this a GOOD or POOR match? (Start with "GOOD MATCH:" or "POOR MATCH:")
2. Why? Consider interest overlap, budget compatibility, and gift-giving potential.
3. Rate compatibility: HIGH, MEDIUM, or LOW"""

            response = openai.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            analysis_text = response.choices[0].message.content
            
            # Determine quality from response
            is_good_match = "GOOD MATCH:" in analysis_text.upper()
            
            # Extract compatibility rating from the end (after "Rating:" or "Compatibility:")
            compatibility = "LOW"  # Default
            if "COMPATIBILITY: HIGH" in analysis_text.upper() or "RATING: HIGH" in analysis_text.upper():
                compatibility = "HIGH"
            elif "COMPATIBILITY: MEDIUM" in analysis_text.upper() or "RATING: MEDIUM" in analysis_text.upper():
                compatibility = "MEDIUM"
            elif "COMPATIBILITY: LOW" in analysis_text.upper() or "RATING: LOW" in analysis_text.upper():
                compatibility = "LOW"
            
            analyses['cognee'].append({
                'giver': giver.name,
                'receiver': receiver.name,
                'analysis': analysis_text,
                'is_good': is_good_match,
                'compatibility': compatibility
            })
            
            progress(0.2 + (0.3 * (i+1) / len(cognee_matches_cache)), 
                    desc=f"Analyzed {i+1}/{len(cognee_matches_cache)} Cognee matches...")
    
    # Analyze LanceDB matches
    if lancedb_matches_cache:
        progress(0.5, desc=f"Analyzing {len(lancedb_matches_cache)} LanceDB matches...")
        
        for i, match in enumerate(lancedb_matches_cache):
            giver = next((p for p in participants if p.id == match.giver_id), None)
            receiver = next((p for p in participants if p.id == match.receiver_id), None)
            
            if not giver or not receiver:
                continue
            
            prompt = f"""Analyze this Secret Santa pairing and determine if it's a good match:

GIVER: {giver.name}
- Interests: {', '.join(giver.interests)}
- Budget: ${giver.budget}
- Preferences: {giver.preferences or 'None'}

RECEIVER: {receiver.name}
- Interests: {', '.join(receiver.interests)}
- Budget: ${receiver.budget}
- Preferences: {receiver.preferences or 'None'}

Matching Method: Pure LanceDB (Vector Similarity Only)

Provide a brief analysis (2-3 sentences):
1. Is this a GOOD or POOR match? (Start with "GOOD MATCH:" or "POOR MATCH:")
2. Why? Consider interest overlap, budget compatibility, and gift-giving potential.
3. Rate compatibility: HIGH, MEDIUM, or LOW"""

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            analysis_text = response.choices[0].message.content
            is_good_match = "GOOD MATCH:" in analysis_text.upper()
            
            # Extract compatibility rating from the end (after "Rating:" or "Compatibility:")
            compatibility = "LOW"  # Default
            if "COMPATIBILITY: HIGH" in analysis_text.upper() or "RATING: HIGH" in analysis_text.upper():
                compatibility = "HIGH"
            elif "COMPATIBILITY: MEDIUM" in analysis_text.upper() or "RATING: MEDIUM" in analysis_text.upper():
                compatibility = "MEDIUM"
            elif "COMPATIBILITY: LOW" in analysis_text.upper() or "RATING: LOW" in analysis_text.upper():
                compatibility = "LOW"
            
            analyses['lancedb'].append({
                'giver': giver.name,
                'receiver': receiver.name,
                'analysis': analysis_text,
                'is_good': is_good_match,
                'compatibility': compatibility
            })
            
            progress(0.5 + (0.4 * (i+1) / len(lancedb_matches_cache)),
                    desc=f"Analyzed {i+1}/{len(lancedb_matches_cache)} LanceDB matches...")
    
    progress(0.95, desc="Formatting results...")
    
    # Format as HTML
    html = format_match_analyses(analyses)
    
    logger.info(f"Completed LLM analysis: {len(analyses['cognee'])} Cognee + {len(analyses['lancedb'])} LanceDB")
    
    return html


def format_match_analyses(analyses):
    """Format LLM match analyses as HTML"""
    
    html = f"""
    <div style='background: #1a1a1a; padding: 20px; border-radius: 10px; max-height: 700px; overflow-y: auto;'>
        <h2 style='color: #667eea; margin-top: 0;'>🤖 LLM Match Quality Analysis</h2>
        <p style='color: #aaa;'>AI-powered evaluation of match compatibility</p>
    """
    
    # Cognee Matches Section
    if analyses['cognee']:
        html += f"""
        <h3 style='color: #667eea; margin-top: 25px; border-bottom: 2px solid #667eea; padding-bottom: 8px;'>
            🎲 Cognee Matches ({len(analyses['cognee'])} pairs)
        </h3>
        """
        
        for analysis in analyses['cognee']:
            # Color based on compatibility level
            if analysis['compatibility'] == 'HIGH':
                border_color = "#55efc4"  # Green
                bg_color = "#1e2a1e"
            elif analysis['compatibility'] == 'MEDIUM':
                border_color = "#fdcb6e"  # Yellow
                bg_color = "#2a2a1e"
            else:  # LOW
                border_color = "#ff6b6b"  # Red
                bg_color = "#2a1e1e"
            
            html += f"""
            <div style='background: {bg_color}; padding: 15px; margin: 12px 0; border-radius: 8px; 
                        border-left: 4px solid {border_color};'>
                <h4 style='color: #e0e0e0; margin: 0 0 8px 0;'>
                    {analysis['giver']} ➡️ {analysis['receiver']}
                    <span style='float: right; color: {border_color}; font-size: 0.9em;'>
                        {analysis['compatibility']} COMPATIBILITY
                    </span>
                </h4>
                <p style='color: #dfe6e9; margin: 0; line-height: 1.6; font-size: 0.95em;'>
                    {analysis['analysis']}
                </p>
            </div>
            """
    
    # LanceDB Matches Section
    if analyses['lancedb']:
        html += f"""
        <h3 style='color: #74b9ff; margin-top: 25px; border-bottom: 2px solid #74b9ff; padding-bottom: 8px;'>
            🔍 LanceDB Matches ({len(analyses['lancedb'])} pairs)
        </h3>
        """
        
        for analysis in analyses['lancedb']:
            # Color based on compatibility level
            if analysis['compatibility'] == 'HIGH':
                border_color = "#55efc4"  # Green
                bg_color = "#1e2a1e"
            elif analysis['compatibility'] == 'MEDIUM':
                border_color = "#fdcb6e"  # Yellow
                bg_color = "#2a2a1e"
            else:  # LOW
                border_color = "#ff6b6b"  # Red
                bg_color = "#2a1e1e"
            
            html += f"""
            <div style='background: {bg_color}; padding: 15px; margin: 12px 0; border-radius: 8px;
                        border-left: 4px solid {border_color};'>
                <h4 style='color: #e0e0e0; margin: 0 0 8px 0;'>
                    {analysis['giver']} ➡️ {analysis['receiver']}
                    <span style='float: right; color: {border_color}; font-size: 0.9em;'>
                        {analysis['compatibility']} COMPATIBILITY
                    </span>
                </h4>
                <p style='color: #dfe6e9; margin: 0; line-height: 1.6; font-size: 0.95em;'>
                    {analysis['analysis']}
                </p>
            </div>
            """
    
    html += "</div>"
    return html

# UI Framework
with gr.Blocks(title="🎅 Secret Santa Agent 🎄", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎅 Secret Santa Agent")
    gr.Markdown("Powered by Cognee Knowledge Graphs")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            include_michael = gr.Checkbox(
                label="Include Michael Scott (Rule Breaker)",
                value=False,
                info="Adds a chaotic participant who loves inappropriate gifts"
            )
            
            load_btn = gr.Button("📂 Load Demo Data", variant="primary", size="lg")
            generate_btn = gr.Button("🎲 Generate Matches", variant="primary", size="lg")
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )
            
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("👥 Participants"):
                    participants_display = gr.HTML(
                        value="No participants loaded. Click 'Load Demo Data' to get started!"
                    )
                
                with gr.Tab("🎲 Cognee Matches"):
                    gr.Markdown("**Hybrid Matching**: Uses Cognee's graph database + vector embeddings for intelligent pairing")
                    matches_display = gr.HTML(
                        value="No matches generated yet. Configure rules and click 'Generate Matches'!"
                    )
                
                with gr.Tab("🔍 Semantic Matches"):
                    gr.Markdown("""
                    ### Pure Vector Similarity Matching
                    This tab uses **only LanceDB** for matching (no Cognee, no Kuzu graph).
                    
                    **How it works:**
                    - Direct vector similarity search on participant embeddings
                    - No graph relationships or multi-hop reasoning
                    - Pure semantic matching based on profile text
                    
                    **Compare with Cognee Matches** to see the difference between:
                    - **Pure vectors** (this tab) vs **Graph + Vectors** (Cognee tab)
                    """)
                    
                    generate_semantic_btn = gr.Button(
                        "🔍 Generate Semantic Matches (LanceDB Only)",
                        variant="secondary",
                        size="lg"
                    )
                    
                    semantic_status = gr.Textbox(
                        label="Semantic Match Status",
                        interactive=False,
                        lines=2
                    )
                    
                    semantic_matches_display = gr.HTML(
                        value="Click 'Generate Semantic Matches' to see pure vector-based matching!"
                    )
                
                
                with gr.Tab("📊 Match Analysis"):
                    gr.Markdown("""
                    ### 🤖 AI-Powered Match Quality Analysis
                    Get LLM evaluation of match compatibility for both Cognee and LanceDB approaches.
                    
                    **Requirements**: Generate both Cognee Matches and Semantic Matches first.
                    """)
                    
                    analyze_matches_btn = gr.Button(
                        "🤖 Analyze Match Quality with LLM",
                        variant="primary",
                        size="lg"
                    )
                    
                    match_analysis_display = gr.HTML(
                        value="Click 'Analyze Match Quality' after generating both types of matches!"
                    )
                
                
                with gr.Tab("🧠 Knowledge Graph"):
                    gr.Markdown("### Cognee Knowledge Graph Visualization")
                    gr.Markdown("This shows the semantic knowledge graph built by Cognee from participant data.")
                    
                    visualize_btn = gr.Button(
                        "🔍 Visualize Knowledge Graph",
                        variant="primary",
                        size="lg"
                    )
                    
                    open_browser_btn = gr.Button(
                        "🌐 Open in Browser",
                        variant="secondary",
                        size="lg",
                        visible=False
                    )
                    
                    graph_status = gr.Textbox(
                        label="Graph Status",
                        interactive=False,
                        lines=2
                    )
                    
                    knowledge_graph_file = gr.File(
                        label="Knowledge Graph HTML File (Download)",
                        file_types=[".html"],
                        type="filepath"
                    )

                with gr.Tab("🎁 GiftScout"):
                    with gr.Row():
                        gift_participant = gr.Dropdown(choices=[], label="Select Participant")
                        search_gifts_btn = gr.Button("🔍 Find Gifts")
                    gift_results = gr.HTML()
                    qna_display = gr.HTML(value=format_documentation_display())

    # Event handlers
    load_btn.click(
        fn=load_demo_data,
        inputs=[include_michael],
        outputs=[status_text, participants_display, gift_participant]
    )
    
    generate_btn.click(
        fn=generate_matches_async,
        inputs=[],
        outputs=[status_text, matches_display]
    )
    
    visualize_btn.click(
        fn=visualize_knowledge_graph,
        outputs=[graph_status, knowledge_graph_file, open_browser_btn]
    )
    
    open_browser_btn.click(
        fn=open_graph_in_browser,
        outputs=[graph_status]
    )
    
    search_gifts_btn.click(search_gift_ideas_cognee, inputs=[gift_participant], outputs=[gift_results])
    
    # Semantic matching (LanceDB only - no Cognee)
    generate_semantic_btn.click(
        fn=generate_semantic_matches_lancedb,
        outputs=[semantic_status, semantic_matches_display]
    )
    
    # LLM Match Analysis
    analyze_matches_btn.click(
        fn=analyze_matches_with_llm,
        outputs=[match_analysis_display]
    )


    gr.Markdown("---")
    gr.Markdown("Made with ❤️ using Gradio • Secret Santa Agent v0.1.0")

if __name__ == "__main__":
    demo.queue().launch()
