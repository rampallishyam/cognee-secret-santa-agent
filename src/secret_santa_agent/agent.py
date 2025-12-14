"""
Secret Santa Agent - Core Logic Module
=======================================
Consolidated module containing the entire agent logic for intelligent
Secret Santa matching powered by Cognee knowledge graphs.

Components:
- Data Models: Participant, Pairing, MatchingRules, BudgetLevel
- LLM Provider: Abstraction for OpenAI/Gemini/Ollama
- Matching Engine: Traditional rule-based matching with scoring
- Gift Suggester: LLM-powered gift recommendations  
- Cognee Integration: Semantic matching using knowledge graphs
- Main Agent: Orchestrator for the complete matching workflow

Key Features:
- Hybrid matching using Cognee (graph + vector search)
- Budget-aware strict matching enforcement
- Reciprocity avoidance (A→B means B≠A)
- Rule breaker mode (Michael Scott) for chaotic mismatches
- LLM-based match reasoning extraction

Author: Secret Santa Agent Team
Version: 0.1.0
"""

import os
import random
import asyncio
import networkx as nx
from enum import Enum
from typing import List, Optional, Dict, Set, Tuple
from datetime import datetime
from collections import defaultdict
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
import cognee
from cognee.modules.search.types import SearchType
from debug_logger import logger, log_section, log_data

# Load environment variables
load_dotenv()

# ==========================================
# 1. DATA MODELS
# ==========================================

class BudgetLevel(str, Enum):
    """Budget levels for gift spending."""
    LOW = "low"  # $20-35
    MEDIUM = "medium"  # $35-60
    HIGH = "high"  # $60-100


class Participant(BaseModel):
    """A Secret Santa participant."""
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    interests: List[str] = Field(default_factory=list, description="List of interests/hobbies")
    budget: int = Field(default=50, ge=20, le=500, description="Budget in dollars")
    budget_level: BudgetLevel = Field(default=BudgetLevel.MEDIUM)
    constraints: List[str] = Field(default_factory=list, description="IDs of people they can't be matched with")
    preferences: Optional[str] = Field(None, description="Gift preferences")
    is_rule_breaker: bool = Field(default=False, description="Michael Scott mode")
    
    def __str__(self) -> str:
        return f"{self.name} ({self.email})"


class Pairing(BaseModel):
    """A Secret Santa pairing."""
    giver_id: str
    receiver_id: str
    year: int = Field(default_factory=lambda: datetime.now().year)
    reasoning: str = Field(default="")
    
    def __str__(self) -> str:
        return f"{self.giver_id} → {self.receiver_id}"


class MatchingRules(BaseModel):
    """Configuration for matching rules."""
    interest_compatibility_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    freshness_years: int = Field(default=3, ge=1, le=5)
    reciprocity_avoidance: bool = Field(default=True)
    min_network_distance: int = Field(default=2, ge=0, le=3)
    wishlist_alignment_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    budget_matching: str = Field(default="flexible")


class GiftSuggestion(BaseModel):
    """An AI-generated gift suggestion."""
    title: str
    description: str
    price_estimate: int
    reasoning: str
    where_to_buy: Optional[str] = None


class Event(BaseModel):
    """A Secret Santa event."""
    year: int
    participants: List[Participant]
    pairings: List[Pairing]
    rules: MatchingRules = Field(default_factory=MatchingRules)
    created_at: datetime = Field(default_factory=datetime.now)


# ==========================================
# 2. LLM PROVIDER
# ==========================================

class LLMProvider:
    """Abstraction layer for LLM providers."""
    
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai")
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        if self.provider == "openai":
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            return openai.OpenAI(api_key=api_key)
        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-pro')
        elif self.provider == "ollama":
            import openai
            return openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            # Fallback for demo purposes if nothing set
            print(f"Warning: Unknown provider {self.provider}, using mock client")
            return None
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        if not self.client:
            return "Mock LLM Response: Great gift idea!"
            
        try:
            if self.provider in ["openai", "ollama"]:
                model = "gpt-3.5-turbo" if self.provider == "openai" else "llama2"
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            elif self.provider == "gemini":
                response = self.client.generate_content(prompt)
                return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"


# ==========================================
# 3. MATCHING ENGINE
# ==========================================

class MatchingEngine:
    """Engine for generating Secret Santa matches based on rules."""
    
    def __init__(self, rules: MatchingRules):
        self.rules = rules
        self.graph = nx.Graph()
    
    def generate_matches(self, participants: List[Participant], past_pairings: List[Pairing]) -> List[Pairing]:
        if len(participants) < 2:
            raise ValueError("Need at least 2 participants")
        
        self._build_graph(participants)
        recent_pairings = self._get_recent_pairings(past_pairings)
        scored_matches = self._score_all_matches(participants, recent_pairings)
        return self._find_optimal_matching(participants, scored_matches, recent_pairings)
    
    def _build_graph(self, participants: List[Participant]):
        self.graph.clear()
        for p in participants:
            self.graph.add_node(p.id, name=p.name)
        
        for i, p1 in enumerate(participants):
            for p2 in participants[i+1:]:
                shared = len(set(p1.interests) & set(p2.interests))
                if shared > 0:
                    self.graph.add_edge(p1.id, p2.id, weight=shared)
    
    def _get_recent_pairings(self, past_pairings: List[Pairing]) -> Dict[str, Set[str]]:
        current_year = datetime.now().year
        cutoff_year = current_year - self.rules.freshness_years
        recent = defaultdict(set)
        for pairing in past_pairings:
            if pairing.year >= cutoff_year:
                recent[pairing.giver_id].add(pairing.receiver_id)
        return recent
    
    def _calculate_interest_compatibility(self, p1: Participant, p2: Participant) -> float:
        if not p1.interests or not p2.interests: return 0.0
        shared = len(set(p1.interests) & set(p2.interests))
        total = len(set(p1.interests) | set(p2.interests))
        if total == 0: return 0.0
        
        jaccard = shared / total
        shared_bonus = min(shared * 10, 30)
        return min(100, jaccard * 100 + shared_bonus)
    
    def _calculate_network_distance(self, giver_id: str, receiver_id: str) -> int:
        try:
            return nx.shortest_path_length(self.graph, giver_id, receiver_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 999
    
    def _score_match(self, giver: Participant, receiver: Participant, recent_pairings: Dict[str, Set[str]]) -> Tuple[float, str]:
        score = 0.0
        reasons = []
        
        # Interest
        interest_score = self._calculate_interest_compatibility(giver, receiver)
        score += interest_score * self.rules.interest_compatibility_weight
        if interest_score > 60: reasons.append(f"High interest compatibility ({interest_score:.0f}%)")
        
        # Freshness
        if receiver.id in recent_pairings.get(giver.id, set()):
            score -= 30
            reasons.append(f"Recent pairing")
        else:
            score += 15
            reasons.append("Fresh pairing")
            
        # Network Distance
        distance = self._calculate_network_distance(giver.id, receiver.id)
        if 999 > distance >= self.rules.min_network_distance:
            score += min(distance * 5, 20)
            reasons.append(f"Good social distance ({distance} hops)")
            
        # Preferences
        if receiver.preferences:
            score += 15 * self.rules.wishlist_alignment_weight
            reasons.append("Has preferences")
            
        # Budget
        if self.rules.budget_matching != "off":
            levels = {"low": 1, "medium": 2, "high": 3}
            if self.rules.budget_matching == "strict":
                compatible = giver.budget_level == receiver.budget_level
            else:
                diff = abs(levels[giver.budget_level.value] - levels[receiver.budget_level.value])
                compatible = diff <= 1
            
            if compatible:
                score += 10
                reasons.append("Budget compatible")
            else:
                score -= 15
                reasons.append("Budget mismatch")
                
        return max(0, min(100, score)), "; ".join(reasons)
    
    def _score_all_matches(self, participants, recent_pairings):
        scores = {}
        for g in participants:
            for r in participants:
                if g.id == r.id or r.id in g.constraints: continue
                scores[(g.id, r.id)] = self._score_match(g, r, recent_pairings)
        return scores
    
    def _find_optimal_matching(self, participants, scored_matches, recent_pairings):
        p_ids = [p.id for p in participants]
        n = len(p_ids)
        sorted_matches = sorted(scored_matches.items(), key=lambda x: x[1][0], reverse=True)
        
        used_givers = set()
        used_receivers = set()
        pairings = []
        
        for (gid, rid), (score, reas) in sorted_matches:
            if gid in used_givers or rid in used_receivers: continue
            
            if self.rules.reciprocity_avoidance:
                if any(p.giver_id == rid and p.receiver_id == gid for p in pairings):
                    continue
            
            pairings.append(Pairing(giver_id=gid, receiver_id=rid, match_score=score, reasoning=reas))
            used_givers.add(gid)
            used_receivers.add(rid)
            if len(pairings) == n: break
            
        # Fallback
        if len(pairings) < n:
            for gid in p_ids:
                if gid not in used_givers:
                    for rid in p_ids:
                        if rid != gid and rid not in used_receivers:
                            sc, re = scored_matches.get((gid, rid), (50.0, "Fallback"))
                            pairings.append(Pairing(giver_id=gid, receiver_id=rid, match_score=sc, reasoning=re))
                            used_receivers.add(rid)
                            break
        return pairings


# ==========================================
# 4. GIFT SUGGESTER
# ==========================================

class GiftSuggester:
    """Generate gift suggestions using LLM."""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
    
    def suggest_gifts(self, giver: Participant, receiver: Participant, num: int = 3) -> List[GiftSuggestion]:
        if giver.is_rule_breaker:
            prompt = self._build_rule_breaker_prompt(giver, receiver, num)
        else:
            prompt = self._build_prompt(giver, receiver, num)
            
        response = self.llm.generate_text(prompt, max_tokens=800)
        return self._parse_suggestions(response, giver.budget)[:num]
    
    def suggest_gifts_with_context(self, giver, receiver, num=3, cognee_insights=None):
        if giver.is_rule_breaker:
            prompt = self._build_rule_breaker_prompt(giver, receiver, num, cognee_insights)
        else:
            prompt = self._build_prompt_with_insights(giver, receiver, num, cognee_insights)
            
        response = self.llm.generate_text(prompt, max_tokens=800)
        return self._parse_suggestions(response, giver.budget)[:num]
        
    def suggest_gifts(self, giver: Participant, receiver: Participant, num: int = 3) -> List[GiftSuggestion]:
        log_section("GENERATING GIFT SUGGESTIONS")
        log_data("Context", f"Giver: {giver.name}, Receiver: {receiver.name}")
        
        if giver.is_rule_breaker:
            prompt = self._build_rule_breaker_prompt(giver, receiver, num)
        else:
            prompt = self._build_prompt(giver, receiver, num)
            
        response = self.llm.generate_text(prompt, max_tokens=800)
        return self._parse_suggestions(response, giver.budget)[:num]

    def _build_prompt(self, giver, receiver, num):
        return f"""Generate {num} gift suggestions for Secret Santa.
Recipient: {receiver.name}
Interests: {', '.join(receiver.interests)}
Preferences: {receiver.preferences}
Budget: ${receiver.budget}

Format: Title, Description, Price, Why, Where (optional).
"""

    def _build_rule_breaker_prompt(self, giver, receiver, num, cognee_insights=None):
        return f"""Generate {num} INAPPROPRIATE gift suggestions from {giver.name} (Rule Breaker).
Context: Giver has ${giver.budget} budget. Receiver ({receiver.name}) likes {receiver.interests}.
Giver ignores budget and social norms. Suggest expensive, mismatched gifts.
"""

    def _build_prompt_with_insights(self, giver, receiver, num, cognee_insights):
        base = self._build_prompt(giver, receiver, num)
        if cognee_insights:
            base += f"\n\nCognee Insights:\n{cognee_insights}\n"
        return base

    def _parse_suggestions(self, response: str, budget: int) -> List[GiftSuggestion]:
        suggestions = []
        # Simple parsing logic
        # In a real app this would be more robust
        if "Title:" in response:
            try:
                # Naive parse for demo
                lines = response.split('\n')
                current_gift = {}
                for line in lines:
                    if line.strip().startswith('Title:'):
                        if current_gift.get('title'):
                            suggestions.append(GiftSuggestion(**current_gift))
                            current_gift = {}
                        current_gift['title'] = line.split(':', 1)[1].strip()
                        current_gift['description'] = "Awesome gift"
                        current_gift['price_estimate'] = budget
                        current_gift['reasoning'] = "Matches interests"
                if current_gift.get('title'):
                    suggestions.append(GiftSuggestion(**current_gift))
            except:
                pass

        if not suggestions:
            suggestions.append(GiftSuggestion(
                title="AI Suggested Gift",
                description=response[:50] + "...",
                price_estimate=budget,
                reasoning="Generated by AI"
            ))
        return suggestions


# ==========================================
# 5. COGNEE INTEGRATION
# ==========================================

class CogneeSemanticMatcher:
    """Uses Cognee knowledge graphs for intelligent matching - PURE COGNEE APPROACH."""
    
    def __init__(self):
        self.initialized = False
        
    async def initialize(self):
        import cognee
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        self.initialized = True
    
    async def index_participants(self, participants: List[Participant]):
        """Index participants into Cognee's knowledge graph."""
        if not self.initialized: 
            await self.initialize()
        
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
        logger.info(f"Indexed {len(participants)} participants into Cognee")
        print(f"✅ Indexed {len(participants)} participants into Cognee")

    async def index_gift_catalog(self, catalog_path: str):
        """Index gift catalog into Cognee knowledge graph."""
        import json
        from pathlib import Path
        
        catalog_file = Path(catalog_path)
        if not catalog_file.exists():
            logger.warning(f"Gift catalog not found at {catalog_path}")
            return False
        
        try:
            with open(catalog_file, 'r') as f:
                catalog = json.load(f)
            
            gift_contexts = []
            for category in catalog['categories']:
                for product in category['products']:
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
            
            await cognee.add(data=gift_contexts, dataset_name="gift_catalog")
            await cognee.cognify(datasets=["gift_catalog"])
            logger.info(f"Indexed {len(gift_contexts)} gifts from {len(catalog['categories'])} categories into Cognee")
            print(f"✅ Indexed {len(gift_contexts)} gift ideas into Cognee")
            return True
        except Exception as e:
            logger.error(f"Error indexing gift catalog: {e}")
            return False
    
    def _extract_match_from_cognee_results(
        self, 
        results, 
        available_receivers: List[Participant],
        giver: Participant
    ) -> Optional[Participant]:
        """Parse Cognee search results to find the best receiver."""
        if not results:
            return None
        
        # Cognee results contain text chunks - look for participant names
        available_names = {p.name.lower(): p for p in available_receivers}
        available_ids = {p.id.lower(): p for p in available_receivers}
        
        # Check each result for mentions of available receivers
        for result in results:
            result_text = str(result).lower()
            
            # Try to find receiver by name
            for name, participant in available_names.items():
                if name in result_text and participant.id != giver.id:
                    return participant
            
            # Try to find receiver by ID
            for pid, participant in available_ids.items():
                if pid in result_text and participant.id != giver.id:
                    return participant
        
        return None

    async def _extract_match_with_llm(
        self,
        results,
        available_receivers: List[Participant],
        giver: Participant
    ) -> Optional[Participant]:
        """Use LLM to extract the best match from Cognee results."""
        if not results:
            return None
        
        log_section("LLM EXTRACTION")
        
        # 1. Prepare Context
        # Flatten Cognee results into a context string
        cognee_context = "\n".join([str(r) for r in results[:4]])
        
        # List valid candidates with FULL PROFILE to prevent hallucination
        candidates_str = "\n".join([
            f"- {p.name} (ID: {p.id})\n  Interests: {', '.join(p.interests)}\n  Preferences: {p.preferences or 'None'}" 
            for p in available_receivers 
            if p.id != giver.id
        ])

        # 2. Construct Prompt
        prompt = f"""Based on the following semantic search results, identify the single best Secret Santa match for {giver.name}.

SEMANTIC SEARCH RESULTS:
{cognee_context}

AVAILABLE CANDIDATES:
{candidates_str}

INSTRUCTIONS:
1. Analyze the search results to find who matches best based on shared interests and compatibility.
2. Select one of the Available Candidates.
3. BE HONEST: If there is a strong match, explain the shared interests.
4. CRITICAL: If NO strong match exists and you are forced to pick from unrelated candidates, ADMIT IT. 
   - State clearly: "Match based primarily on availability; interests (X vs Y) are distinct."
   - Do NOT invent connections (e.g., do NOT say "Astronomy matches Gaming").
5. Return the result as a strict JSON object with two fields: "id" and "reasoning".

EXAMPLE OUTPUT (Strong Match):
{{
  "id": "p_123",
  "reasoning": "Emma is the best match because her interest in sustainability perfectly aligns with Alex's passion for eco-friendly gardening."
}}

EXAMPLE OUTPUT (Weak Match):
{{
  "id": "p_456",
  "reasoning": "Selected based on availability. Compatibility is low as giver likes Sports and receiver likes Knitting, but no better options remain."
}}
"""
        log_data("INPUT PROMPT", prompt)

        # 3. Call LLM
        try:
            llm = LLMProvider() 
            response_text = llm.generate_text(prompt, max_tokens=150).strip()
            
            log_data("RAW OUTPUT", response_text)
            
            # 4. Parse JSON Response
            import json
            import re
            
            match_id = None
            reasoning = "Matched via Cognee Semantic Search"
            
            try:
                # Try to find JSON block if wrapped in markdown
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    match_id = data.get("id")
                    reasoning = data.get("reasoning", reasoning)
                else:
                    # Fallback: Try straight JSON load
                    data = json.loads(response_text)
                    match_id = data.get("id")
                    reasoning = data.get("reasoning", reasoning)
                    
            except json.JSONDecodeError:
                # Fallback: unexpected format, try simple string extraction if possible
                print("JSON parse failed, falling back to raw ID cleaner")
                match_id = response_text.replace("'", "").replace('"', "").replace(".", "")
                
            
            if match_id:
                clean_id = match_id.replace("'", "").replace('"', "").replace(".", "")
                for p in available_receivers:
                    if p.id == clean_id:
                        print(f"LLM extracted match: {p.name} ({p.id})")
                        log_data("FINAL MATCH", f"{p.name} ({p.id})")
                        log_data("REASONING", reasoning)
                        
                        # Attach reasoning to participant temporarily or return tuple
                        # For this function signature, we're returning Participant, but we need to return reasoning too.
                        # We will attach it as a temporary attribute to the participant object to avoid breaking signature heavily
                        p._temp_reasoning = reasoning
                        return p

        except Exception as e:
            print(f"LLM Extraction failed: {e}")
            logger.error(f"LLM Extraction failed: {e}")
            
        return None

    async def find_best_match(
        self, 
        giver: Participant, 
        available_receivers: List[Participant],
        past_pairings: List[Pairing] = None,
        rules: MatchingRules = None
    ) -> Tuple[Participant, str]:
        """Find best match using PURE Cognee semantic search - no scoring."""
        
        # Build semantic query for Cognee
        interests_str = ", ".join(giver.interests[:5])
        prefs_str = f" They prefer {giver.preferences}." if giver.preferences else ""
        
        query = f"""Find the best Secret Santa gift recipient match for {giver.name}.
{giver.name} enjoys: {interests_str}.{prefs_str}
Who would be most compatible based on shared or complementary interests?
Consider personality fit and gift compatibility."""
        
        try:
            log_section("COGNEE HYBRID SEARCH")
            log_data("QUERY", query)
            
            # Query Cognee's knowledge graph using Hybrid Retrieval (Vector + Graph)
            results = await cognee.search(
                query_text=query, 
                query_type=SearchType.GRAPH_COMPLETION
            )
            
            # Debug: See what hybrid search found
            # print(f"DEBUG Hybrid results: {results}")
            
            formatted_results = "\n".join([f"[{i+1}] {str(r)[:300]}..." for i,r in enumerate(results[:3])])
            log_data("COGNEE OUTPUT (Vector + Graph)", formatted_results)

            # Parse Cognee's results using LLM for robust extraction
            best_match = await self._extract_match_with_llm(results, available_receivers, giver)
            
            if best_match:
                # Use the reasoning extracted from the LLM if available
                reasoning = getattr(best_match, "_temp_reasoning", f"Cognee Hybrid Graph Retrieval identified {best_match.name} implies deep compatibility.")
                # Clean up temporary attribute
                if hasattr(best_match, "_temp_reasoning"):
                    del best_match._temp_reasoning
                    
                return best_match, reasoning
            
        except Exception as e:
            print(f"Cognee search error for {giver.name}: {e}")
        
        # Fallback: if Cognee didn't find a match, pick the receiver with most shared interests
        best_receiver = max(
            available_receivers,
            key=lambda r: len(set(giver.interests) & set(r.interests))
        )
        
        return best_receiver, "Matched using interest overlap (Cognee fallback)"

    async def find_worst_match(
        self, 
        giver: Participant, 
        receivers: List[Participant],
        past_pairings: List[Pairing] = None,
        rules: MatchingRules = None
    ) -> Tuple[Participant, str]:
        """Find worst match for rule breakers using Cognee."""
        
        interests_str = ", ".join(giver.interests[:5])
        
        query = f"""Find the LEAST compatible Secret Santa match for {giver.name}.
{giver.name} likes: {interests_str}.
Who would be the most incompatible or have completely different interests?
Find someone who would create an awkward or mismatched pairing."""
        
        try:
            results = await cognee.search(query_text=query,query_type=SearchType.GRAPH_COMPLETION)
            worst_match = self._extract_match_from_cognee_results(results, receivers, giver)
            
            if worst_match:
                return worst_match, f"Cognee found {worst_match.name} as intentionally incompatible (rule breaker mode)"
            
        except Exception as e:
            print(f"Cognee search error for rule breaker {giver.name}: {e}")
        
        # Fallback: pick someone with least shared interests
        worst_receiver = min(
            receivers,
            key=lambda r: len(set(giver.interests) & set(r.interests))
        )
        
        return worst_receiver, "Mismatched pairing (rule breaker mode)"

    async def get_gift_insights(self, receiver, budget) -> Dict:
        """Query Cognee for gift ideas."""
        try:
            query = f"Suggest thoughtful gift ideas for someone who likes {', '.join(receiver.interests[:3])} within a ${budget} budget"
            results = await cognee.search(query_text=query)
            return {"cognee_suggestions": [str(r) for r in results[:3]]}
        except:
            return {}



# ==========================================
# 6. MAIN AGENT
# ==========================================

class SecretSantaAgent:
    """Main agent for Secret Santa matching."""
    
    def __init__(self, llm_provider: Optional[str] = None):
        self.llm = LLMProvider(llm_provider) if llm_provider else None
        self.participants: List[Participant] = []
        self.past_pairings: List[Pairing] = []
        self.rules = MatchingRules()
        self.matching_engine = MatchingEngine(self.rules)
        self.gift_suggester = GiftSuggester(self.llm) if self.llm else None
        self.cognee_matcher = CogneeSemanticMatcher()
        self.knowledge_built = False
        log_section("AGENT INITIALIZATION")
        logger.info(f"LLM Provider: {llm_provider if llm_provider else 'Default'}")
        
    def add_participant(self, p):
        self.participants.append(p)
        
    def update_rules(self, r): 
        self.rules = r; 
        self.matching_engine = MatchingEngine(r)
        logger.info(f"Rules updated: {r}")
    
    async def build_knowledge_graph(self):
        log_section("BUILDING KNOWLEDGE GRAPH")
        if self.participants:
            await self.cognee_matcher.index_participants(self.participants)
            self.knowledge_built = True
            logger.info("Knowledge Graph building complete.")
            
    async def generate_matches_with_cognee(self) -> List[Pairing]:
        if not self.knowledge_built: await self.build_knowledge_graph()
        
        matches = []
        used_receivers = set()
        shuffled = list(self.participants)
        random.shuffle(shuffled)
        
        for idx, giver in enumerate(shuffled):
            available = [p for p in self.participants if p.id not in used_receivers and p.id != giver.id and p.id not in giver.constraints]
            
            # Last giver cycle closing logic
            # Simplified: if no available, just pick random (should implement cycle closing)
            if not available:
                 # Try to steal a receiver (basic swap) or fail gracefully
                 print(f"Warning: No receivers for {giver.name}")
                 continue
            
            if giver.is_rule_breaker:
                rec, reas = await self.cognee_matcher.find_worst_match(
                    giver, available, self.past_pairings, self.rules
                )
            else:
                rec, reas = await self.cognee_matcher.find_best_match(
                    giver, available, self.past_pairings, self.rules
                )
                
            matches.append(Pairing(giver_id=giver.id, receiver_id=rec.id, reasoning=reas))
            used_receivers.add(rec.id)
            
        return matches

    def generate_matches(self):
        return asyncio.run(self.generate_matches_with_cognee())
    
    def get_statistics(self) -> Dict:
        return {
            "total_participants": len(self.participants),
            "cognee_enabled": self.knowledge_built
        }

