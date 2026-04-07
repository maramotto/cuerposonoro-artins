import pytest
from audio.chords import Chord, ChordProgression


class TestChord:
    def test_chord_has_name_root_and_notes(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        assert chord.name == "Dm9"
        assert chord.root == 62
        assert chord.notes == [62, 65, 69, 72, 76]

    def test_nearest_note_exact_match(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        assert chord.nearest_note(62) == 62

    def test_nearest_note_between_two(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        # 63 (Eb) is closer to 62 (D) than 65 (F)
        assert chord.nearest_note(63) == 62
        # 66 (F#) is closer to 65 (F) than 69 (A)
        assert chord.nearest_note(66) == 65

    def test_nearest_note_respects_range(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        # Asking for note in a higher octave should find the transposed note
        result = chord.nearest_note(86)
        # 86 is D6 (62+24), should find a chord tone near 86
        assert result in chord.all_notes_in_range(48, 96)

    def test_all_notes_in_range(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        notes = chord.all_notes_in_range(48, 84)
        # All returned notes must be in range
        assert all(48 <= n <= 84 for n in notes)
        # Must contain the original notes
        for n in chord.notes:
            if 48 <= n <= 84:
                assert n in notes
        # Must be sorted
        assert notes == sorted(notes)

    def test_all_notes_in_range_includes_octave_transpositions(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        notes = chord.all_notes_in_range(48, 84)
        # D3=50 should be present (62-12)
        assert 50 in notes
        # F3=53 should be present (65-12)
        assert 53 in notes

    def test_note_from_height_low(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        note = chord.note_from_height(0.0, note_min=48, note_max=84)
        assert note == chord.nearest_note(48)

    def test_note_from_height_high(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        note = chord.note_from_height(1.0, note_min=48, note_max=84)
        assert note == chord.nearest_note(84)

    def test_note_from_height_mid(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        note = chord.note_from_height(0.5, note_min=48, note_max=84)
        mid_midi = 48 + 0.5 * (84 - 48)
        assert note == chord.nearest_note(int(mid_midi))

    def test_nearest_note_fallback_to_full_range(self):
        """When no candidates in ±12 range, fall back to full 0-127 search."""
        # A chord with only very high notes — target at MIDI 10 has nothing ±12
        chord = Chord(name="HighOnly", root=120, notes=[120])
        result = chord.nearest_note(10)
        # Pitch class of 120 is 0 (C). Nearest C to 10 is 12 (C1).
        assert result == 12


class TestChordTension:
    """Test head-tilt chord tension modification."""

    def test_chord_has_simplified_and_tension_variants(self):
        chord = Chord(
            name="Dm9", root=62,
            notes=[62, 65, 69, 72, 76],
            simplified=[62, 65, 69],          # Dm triad
            tension=[62, 65, 69, 72, 76, 77], # Dm9 + 11th (G)
        )
        assert chord.simplified == [62, 65, 69]
        assert chord.tension == [62, 65, 69, 72, 76, 77]

    def test_default_simplified_and_tension_are_notes(self):
        chord = Chord(name="Dm9", root=62, notes=[62, 65, 69, 72, 76])
        assert chord.simplified == [62, 65, 69, 72, 76]
        assert chord.tension == [62, 65, 69, 72, 76]

    def test_active_notes_at_zero_tilt_returns_base(self):
        chord = Chord(
            name="Dm9", root=62,
            notes=[62, 65, 69, 72, 76],
            simplified=[62, 65, 69],
            tension=[62, 65, 69, 72, 76, 77],
        )
        active = chord.active_notes(tilt=0.0)
        assert active == [62, 65, 69, 72, 76]

    def test_active_notes_positive_tilt_returns_tension(self):
        chord = Chord(
            name="Dm9", root=62,
            notes=[62, 65, 69, 72, 76],
            simplified=[62, 65, 69],
            tension=[62, 65, 69, 72, 76, 77],
        )
        active = chord.active_notes(tilt=1.0)
        assert active == [62, 65, 69, 72, 76, 77]

    def test_active_notes_negative_tilt_returns_simplified(self):
        chord = Chord(
            name="Dm9", root=62,
            notes=[62, 65, 69, 72, 76],
            simplified=[62, 65, 69],
            tension=[62, 65, 69, 72, 76, 77],
        )
        active = chord.active_notes(tilt=-1.0)
        assert active == [62, 65, 69]

    def test_note_from_height_uses_active_notes(self):
        chord = Chord(
            name="Dm9", root=62,
            notes=[62, 65, 69, 72, 76],
            simplified=[62, 65, 69],
            tension=[62, 65, 69, 72, 76, 77],
        )
        # With full tension, 77 (F4) should be reachable
        note = chord.note_from_height(0.5, 48, 84, tilt=1.0)
        all_tension_in_range = chord.all_notes_in_range(48, 84, tilt=1.0)
        assert note in all_tension_in_range


class TestChordProgression:
    @pytest.fixture
    def progression(self):
        chords = [
            Chord("Dm9", 62, [62, 65, 69, 72, 76]),
            Chord("G13sus4", 67, [67, 72, 74, 76, 81]),
            Chord("Cmaj7#11", 60, [60, 64, 67, 71, 78]),
            Chord("Fmaj9", 65, [65, 69, 72, 76, 79]),
            Chord("Bø7", 71, [71, 74, 77, 81]),
            Chord("E7alt", 64, [64, 68, 70, 75, 78]),
        ]
        return ChordProgression(chords)

    def test_starts_at_first_chord(self, progression):
        assert progression.current.name == "Dm9"
        assert progression.index == 0

    def test_advance(self, progression):
        progression.advance()
        assert progression.current.name == "G13sus4"
        assert progression.index == 1

    def test_retreat(self, progression):
        progression.advance()
        progression.advance()
        progression.retreat()
        assert progression.current.name == "G13sus4"

    def test_retreat_at_start_stays(self, progression):
        progression.retreat()
        assert progression.index == 0

    def test_advance_wraps_around(self, progression):
        for _ in range(6):
            progression.advance()
        assert progression.index == 0
        assert progression.current.name == "Dm9"

    def test_from_config(self):
        config = [
            {"name": "Dm9", "root": 62, "notes": [62, 65, 69, 72, 76]},
            {"name": "G13sus4", "root": 67, "notes": [67, 72, 74, 76, 81]},
        ]
        prog = ChordProgression.from_config(config)
        assert len(prog.chords) == 2
        assert prog.current.name == "Dm9"
