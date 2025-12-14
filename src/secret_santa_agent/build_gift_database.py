"""
Build Gift Database for Secret Santa Agent
==========================================
This script creates a pre-built Cognee knowledge graph from the gift catalog
using the proper export helper for correct directory structure.
"""

import asyncio
import os
import json
import cognee
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env variables (for API keys)
load_dotenv()

# Add parent directory to path to import helpers
sys.path.append(str(Path(__file__).parent))

from helper_functions.export_cognee import export_cognee_data
from cognee.api.v1.visualize.visualize import visualize_graph


async def build_gift_database():
    """Build and export the gift catalog knowledge graph."""
    
    print("🎁 Starting creation of Pre-built Gift Database...")
    
    # 1. Initialize Cognee (prune existing data)
    print("\n[1/4] Pruning existing Cognee data...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    
    # 2. Load Gift Catalog
    print("\n[2/4] Loading gift catalog...")
    catalog_path = Path(__file__).parent.parent.parent / "data" / "gift_catalog.json"
    
    with open(catalog_path, "r") as f:
        catalog = json.load(f)
    
    # 3. Index gifts into Cognee
    print("\n[3/4] Indexing gifts into Knowledge Graph...")
    gift_contexts = []
    
    for category in catalog['categories']:
        for product in category['products']:
            # Create rich context for each gift
            context = f"""Gift Product:
Name: {product['name']}
Category: {category['name']}
Price: ${product['price']}
Best for interests: {', '.join(category['interests'])}
Description: {product['description']}
Budget Range: {'Low ($20-$40)' if product['price'] < 40 else 'Medium ($40-$70)' if product['price'] < 70 else 'High ($70+)'}
Perfect gift for someone who enjoys: {', '.join(category['interests'][:3])}
"""
            gift_contexts.append(context)
    
    print(f"  Found {len(gift_contexts)} gifts across {len(catalog['categories'])} categories")
    
    await cognee.add(data=gift_contexts, dataset_name="gift_catalog")
    await cognee.cognify(datasets=["gift_catalog"])
    print("  ✓ Gifts indexed and graph built.")
    
    # 4. Export Database using proper helper
    print("\n[4/4] Exporting database for runtime use...")
    
    output_path = Path(__file__).parent / "data" / "cognee_gifts_db"
    
    # Make sure parent directory exists
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Remove old v2 if exists
    import shutil
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Use helper to export (creates proper structure with system_databases/)
    await export_cognee_data(export_dir=str(output_path))

    # Create database visualisation file
    await visualize_graph(os.getcwd() + "/artifacts/knowledge_graph_gifts.html")
    
    print("\n✅ Gift Database creation complete!")
    print(f"   Saved to: {output_path}")
    print(f"   Structure: {output_path}/system_databases/")


if __name__ == "__main__":
    asyncio.run(build_gift_database())
