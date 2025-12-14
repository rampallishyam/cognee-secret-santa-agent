
import asyncio
import json
import cognee
from cognee.api.v1.search import SearchType
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env variables (for API keys)
load_dotenv()

# Add parent directory to path to import helpers
sys.path.append(str(Path(__file__).parent))

from helper_functions.export_cognee import export_cognee_data
from agent import Participant, BudgetLevel

async def build_participant_database():
    """Butild and export the full participant knowledge graph."""
    
    print("🚀 Starting creation of Pre-built Participant Database...")
    
    # 1. Initialize Cognee
    print("\n[1/4] specialized pruning (resetting DB)...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    
    # 2. Load Participants (INCLUDING MICHAEL)
    print("\n[2/4] Loading participant data...")
    data_path = Path(__file__).parent.parent.parent / "data" / "participants_with_michael.json"
    
    with open(data_path, "r") as f:
        data = json.load(f)
        
    participants = []
    for p_data in data:
        # Convert dictionary to Participant object
        # Enum handling
        if "budget_level" in p_data:
            p_data["budget_level"] = BudgetLevel(p_data["budget_level"])
            
        participants.append(Participant(**p_data))
        
    print(f"  Loaded {len(participants)} participants.")

    # 3. Index into Cognee
    print("\n[3/4] Indexing participants into Knowledge Graph...")
    texts = []
    for p in participants:
        # Create rich context for Cognee
        context = f"""Participant Profile:
Name: {p.name}
ID: {p.id}
Email: {p.email}
Interests and Hobbies: {', '.join(p.interests)}
Gift Preferences: {p.preferences or 'No specific preferences'}
Budget Level: {p.budget_level.value} (${p.budget})
Special Notes: {'Rule breaker - likes chaos and inappropriate gifts' if p.is_rule_breaker else 'Standard participant'}
"""
        texts.append(context)
        
    await cognee.add(data=texts, dataset_name="secret_santa_participants")
    await cognee.cognify(datasets=["secret_santa_participants"])
    print("  ✓ Participants indexed and graph built.")
    
    # 4. Export Database
    print("\n[4/4] Exporting database for runtime use...")
    
    output_path = Path(__file__).parent / "data" / "cognee_participants_db"
    
    # Make sure parent directory exists
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Use helper to export
    await export_cognee_data(export_dir=str(output_path))
    
    print("\n✅ Database creation complete!")
    print(f"   Saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(build_participant_database())
