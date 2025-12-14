"""Unit tests for the Secret Santa matching engine."""

import pytest
from datetime import datetime

from secret_santa_agent import (
    Participant,
    Pairing,
    MatchingRules,
    BudgetLevel,
    MatchingEngine,
)


# Fixtures
@pytest.fixture
def sample_participants():
    """Create a set of sample participants for testing."""
    return [
        Participant(
            id="p1",
            name="Alice",
            email="alice@test.com",
            interests=["coding", "gaming", "sci-fi"],
            budget=50,
            budget_level=BudgetLevel.MEDIUM,
        ),
        Participant(
            id="p2",
            name="Bob",
            email="bob@test.com",
            interests=["cooking", "travel", "music"],
            budget=50,
            budget_level=BudgetLevel.MEDIUM,
        ),
        Participant(
            id="p3",
            name="Charlie",
            email="charlie@test.com",
            interests=["coding", "gaming", "music"],
            budget=75,
            budget_level=BudgetLevel.HIGH,
        ),
        Participant(
            id="p4",
            name="Diana",
            email="diana@test.com",
            interests=["reading", "travel", "yoga"],
            budget=30,
            budget_level=BudgetLevel.LOW,
        ),
    ]


@pytest.fixture
def sample_past_pairings():
    """Create sample past pairings."""
    return [
        Pairing(giver_id="p1", receiver_id="p2", year=2023, match_score=85.0),
        Pairing(giver_id="p2", receiver_id="p3", year=2023, match_score=80.0),
        Pairing(giver_id="p3", receiver_id="p4", year=2023, match_score=75.0),
        Pairing(giver_id="p4", receiver_id="p1", year=2023, match_score=78.0),
    ]


@pytest.fixture
def default_rules():
    """Create default matching rules."""
    return MatchingRules()


@pytest.fixture
def engine(default_rules):
    """Create a matching engine instance."""
    return MatchingEngine(default_rules)


# Tests for MatchingEngine
class TestMatchingEngine:
    """Tests for the MatchingEngine class."""

    def test_generate_matches_basic(self, engine, sample_participants):
        """Test basic match generation."""
        matches = engine.generate_matches(sample_participants, [])
        
        # Should have one match per participant
        assert len(matches) == len(sample_participants)
        
        # Each participant should be a giver exactly once
        givers = [m.giver_id for m in matches]
        assert len(set(givers)) == len(sample_participants)
        
        # Each participant should be a receiver exactly once
        receivers = [m.receiver_id for m in matches]
        assert len(set(receivers)) == len(sample_participants)

    def test_no_self_matching(self, engine, sample_participants):
        """Test that no one is matched to themselves."""
        matches = engine.generate_matches(sample_participants, [])
        
        for match in matches:
            assert match.giver_id != match.receiver_id

    def test_constraint_respected(self, sample_participants, default_rules):
        """Test that constraints are respected."""
        # Add constraint: p1 cannot match with p2
        sample_participants[0].constraints = ["p2"]
        
        engine = MatchingEngine(default_rules)
        matches = engine.generate_matches(sample_participants, [])
        
        # p1 should not give to p2
        p1_match = next(m for m in matches if m.giver_id == "p1")
        assert p1_match.receiver_id != "p2"

    def test_reciprocity_avoidance(self, sample_participants, default_rules):
        """Test that reciprocal pairings are avoided in the same year."""
        rules = MatchingRules(reciprocity_avoidance=True)
        engine = MatchingEngine(rules)
        
        matches = engine.generate_matches(sample_participants, [])
        
        # Check no A->B and B->A pairs
        pairs = {(m.giver_id, m.receiver_id) for m in matches}
        for giver_id, receiver_id in pairs:
            assert (receiver_id, giver_id) not in pairs

    def test_interest_compatibility_scoring(self, engine, sample_participants):
        """Test interest compatibility scoring."""
        # p1 and p3 share interests (coding, gaming)
        p1 = sample_participants[0]
        p3 = sample_participants[2]
        
        score = engine._calculate_interest_compatibility(p1, p3)
        
        # Should have positive compatibility score
        assert score > 0
        # Should be higher than 50 (neutral) due to shared interests
        assert score > 50

    def test_interest_compatibility_no_overlap(self, engine, sample_participants):
        """Test interest compatibility with no overlapping interests."""
        p1 = sample_participants[0]  # coding, gaming, sci-fi
        p4 = sample_participants[3]  # reading, travel, yoga
        
        score = engine._calculate_interest_compatibility(p1, p4)
        
        # Should be 0 due to no overlap
        assert score == 0

    def test_freshness_factor(self, engine, sample_participants, sample_past_pairings):
        """Test that freshness factor affects scoring."""
        matches_with_history = engine.generate_matches(
            sample_participants, sample_past_pairings
        )
        matches_without_history = engine.generate_matches(sample_participants, [])
        
        # Both should generate valid matches
        assert len(matches_with_history) == len(sample_participants)
        assert len(matches_without_history) == len(sample_participants)

    def test_minimum_participants(self, engine):
        """Test error handling with too few participants."""
        single_participant = [
            Participant(
                id="p1",
                name="Solo",
                email="solo@test.com",
            )
        ]
        
        with pytest.raises(ValueError, match="at least 2"):
            engine.generate_matches(single_participant, [])

    def test_budget_compatibility_strict(self, sample_participants):
        """Test strict budget matching."""
        rules = MatchingRules(budget_matching="strict")
        engine = MatchingEngine(rules)
        
        p1 = sample_participants[0]  # MEDIUM
        p3 = sample_participants[2]  # HIGH
        
        is_compatible = engine._check_budget_compatibility(p1, p3)
        assert is_compatible is False  # Different levels

    def test_budget_compatibility_flexible(self, sample_participants):
        """Test flexible budget matching."""
        rules = MatchingRules(budget_matching="flexible")
        engine = MatchingEngine(rules)
        
        p1 = sample_participants[0]  # MEDIUM
        p3 = sample_participants[2]  # HIGH
        
        is_compatible = engine._check_budget_compatibility(p1, p3)
        assert is_compatible is True  # One level difference allowed

    def test_budget_compatibility_off(self, sample_participants):
        """Test budget matching disabled."""
        rules = MatchingRules(budget_matching="off")
        engine = MatchingEngine(rules)
        
        p1 = sample_participants[0]  # MEDIUM
        p4 = sample_participants[3]  # LOW
        
        is_compatible = engine._check_budget_compatibility(p1, p4)
        assert is_compatible is True  # Always compatible when off


class TestMatchScoring:
    """Tests for match scoring functionality."""

    def test_match_score_range(self, engine, sample_participants):
        """Test that match scores are in valid range."""
        matches = engine.generate_matches(sample_participants, [])
        
        for match in matches:
            assert 0 <= match.match_score <= 100

    def test_match_has_reasoning(self, engine, sample_participants):
        """Test that matches include reasoning."""
        matches = engine.generate_matches(sample_participants, [])
        
        for match in matches:
            assert match.reasoning  # Should have some reasoning text

    def test_high_compatibility_bonus(self, engine, sample_participants):
        """Test that high interest compatibility increases score."""
        # p1 and p3 have high compatibility
        scores = engine._score_all_matches(sample_participants, {})
        
        # Get score for p1 -> p3
        score_p1_p3 = scores.get(("p1", "p3"), (0, ""))
        # Get score for p1 -> p4 (low compatibility)
        score_p1_p4 = scores.get(("p1", "p4"), (0, ""))
        
        # p1->p3 should score higher due to shared interests
        assert score_p1_p3[0] > score_p1_p4[0]


# Run tests with: uv run pytest tests/test_matching.py -v
