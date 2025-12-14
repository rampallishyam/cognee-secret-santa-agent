"""Unit tests for the Secret Santa Agent."""

import pytest
from unittest.mock import Mock, patch

from src.secret_santa_agent import (
    SecretSantaAgent,
    Participant,
    Pairing,
    MatchingRules,
    BudgetLevel,
)


@pytest.fixture
def mock_agent():
    """Create an agent without LLM initialization."""
    with patch.object(SecretSantaAgent, '__init__', lambda self, *args: None):
        agent = SecretSantaAgent.__new__(SecretSantaAgent)
        agent.participants = []
        agent.past_pairings = []
        agent.rules = MatchingRules()
        from src.secret_santa_agent import MatchingEngine
        agent.matching_engine = MatchingEngine(agent.rules)
        agent.knowledge_built = False
        agent.llm = None
        agent.gift_suggester = None
        return agent


@pytest.fixture
def sample_participants():
    """Create sample participants."""
    return [
        Participant(
            id="p1",
            name="Alice",
            email="alice@test.com",
            interests=["coding", "gaming"],
            budget=50,
            budget_level=BudgetLevel.MEDIUM,
        ),
        Participant(
            id="p2",
            name="Bob",
            email="bob@test.com",
            interests=["cooking", "travel"],
            budget=60,
            budget_level=BudgetLevel.MEDIUM,
        ),
        Participant(
            id="p3",
            name="Charlie",
            email="charlie@test.com",
            interests=["music", "art"],
            budget=40,
            budget_level=BudgetLevel.MEDIUM,
        ),
    ]


class TestSecretSantaAgent:
    """Tests for SecretSantaAgent class."""

    def test_add_participant(self, mock_agent, sample_participants):
        """Test adding a single participant."""
        mock_agent.add_participant(sample_participants[0])
        
        assert len(mock_agent.participants) == 1
        assert mock_agent.participants[0].name == "Alice"

    def test_add_multiple_participants(self, mock_agent, sample_participants):
        """Test adding multiple participants at once."""
        mock_agent.add_participants(sample_participants)
        
        assert len(mock_agent.participants) == 3

    def test_set_past_pairings(self, mock_agent):
        """Test setting past pairings."""
        pairings = [
            Pairing(giver_id="p1", receiver_id="p2", year=2023),
            Pairing(giver_id="p2", receiver_id="p1", year=2023),
        ]
        mock_agent.set_past_pairings(pairings)
        
        assert len(mock_agent.past_pairings) == 2

    def test_update_rules(self, mock_agent):
        """Test updating matching rules."""
        new_rules = MatchingRules(
            interest_compatibility_weight=0.9,
            freshness_years=5,
            reciprocity_avoidance=False,
        )
        mock_agent.update_rules(new_rules)
        
        assert mock_agent.rules.interest_compatibility_weight == 0.9
        assert mock_agent.rules.freshness_years == 5
        assert mock_agent.rules.reciprocity_avoidance is False

    def test_generate_matches(self, mock_agent, sample_participants):
        """Test match generation."""
        mock_agent.add_participants(sample_participants)
        matches = mock_agent.generate_matches()
        
        assert len(matches) == 3
        
        # Each person gives once
        givers = {m.giver_id for m in matches}
        assert givers == {"p1", "p2", "p3"}
        
        # Each person receives once
        receivers = {m.receiver_id for m in matches}
        assert receivers == {"p1", "p2", "p3"}

    def test_generate_matches_no_participants(self, mock_agent):
        """Test error when generating matches with no participants."""
        with pytest.raises(ValueError, match="No participants"):
            mock_agent.generate_matches()

    def test_get_participant_by_id(self, mock_agent, sample_participants):
        """Test retrieving participant by ID."""
        mock_agent.add_participants(sample_participants)
        
        participant = mock_agent.get_participant_by_id("p2")
        assert participant is not None
        assert participant.name == "Bob"

    def test_get_participant_by_id_not_found(self, mock_agent, sample_participants):
        """Test retrieving non-existent participant."""
        mock_agent.add_participants(sample_participants)
        
        participant = mock_agent.get_participant_by_id("p999")
        assert participant is None

    def test_get_statistics(self, mock_agent, sample_participants):
        """Test statistics calculation."""
        mock_agent.add_participants(sample_participants)
        mock_agent.past_pairings = [
            Pairing(giver_id="p1", receiver_id="p2", year=2023),
        ]
        
        stats = mock_agent.get_statistics()
        
        assert stats["total_participants"] == 3
        assert stats["total_past_pairings"] == 1
        assert stats["average_budget"] == 50  # (50+60+40)/3
        assert stats["total_interests"] > 0

    def test_get_statistics_empty(self, mock_agent):
        """Test statistics with no data."""
        stats = mock_agent.get_statistics()
        
        assert stats["total_participants"] == 0
        assert stats["total_past_pairings"] == 0
        assert stats["average_budget"] == 0
        assert stats["total_interests"] == 0


class TestModels:
    """Tests for data models."""

    def test_participant_str(self):
        """Test participant string representation."""
        p = Participant(id="p1", name="Alice", email="alice@test.com")
        assert str(p) == "Alice (alice@test.com)"

    def test_pairing_str(self):
        """Test pairing string representation."""
        p = Pairing(giver_id="p1", receiver_id="p2", match_score=85.5)
        assert "p1 →" in str(p)
        assert "p2" in str(p)
        assert "85.5" in str(p)

    def test_participant_defaults(self):
        """Test participant default values."""
        p = Participant(id="p1", name="Test", email="test@test.com")
        
        assert p.interests == []
        assert p.budget == 50
        assert p.budget_level == BudgetLevel.MEDIUM
        assert p.constraints == []
        assert p.preferences is None

    def test_matching_rules_defaults(self):
        """Test matching rules default values."""
        rules = MatchingRules()
        
        assert rules.interest_compatibility_weight == 0.8
        assert rules.freshness_years == 3
        assert rules.reciprocity_avoidance is True
        assert rules.min_network_distance == 2
        assert rules.wishlist_alignment_weight == 0.7
        assert rules.budget_matching == "flexible"


# Run tests with: uv run pytest tests/test_agent.py -v
