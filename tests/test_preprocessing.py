import pytest

from nya_ir.preprocessing import (
    RuleBasedNyaResolver,
    SuffixNyaRemover,
    preprocess_keep,
    preprocess_naive_strip,
    preprocess_sastrawi_clitic,
    preprocess_sentinel,
)


FALSE_POSITIVES = ["punya", "tanya", "hanya", "biasanya", "Kenya", "Sonya", "Tanya"]
KNOWN_ROOT_TOKENS = ["rumahnya", "bukunya", "pidatonya"]
SMALL_TEST_ROOTS = {"rumah", "buku", "pidato"}


class _EmptyReturningRemover:
    """Test double that simulates the Sastrawi-visitor edge case where remove() returns ''."""

    def filter(self, token: str) -> str:  # noqa: D401 - protocol method
        return ""


def test_resolver_skips_occurrences_with_empty_root():
    """P1 fix: Sastrawi's visitor can return '' on edge cases; resolver must not emit
    a malformed ' antecedent' substitution. The whitespace-empty guard in detect() must
    short-circuit before the dictionary check, otherwise '' ends up substituted with a
    leading-space artefact (silent corpus corruption on full-corpus runs)."""
    resolver = RuleBasedNyaResolver(
        root_dict={"pidato", "", "rumah"},  # include "" to prove guard fires before dict check
        remover=_EmptyReturningRemover(),
    )
    text = "Soekarno datang. Pidatonya menggemparkan."
    # Pre-fix: resolver would substitute "Pidatonya" with " Soekarno" (leading space, empty root).
    # Post-fix: empty root short-circuits → no substitution → text unchanged.
    assert resolver.resolve(text) == text


def test_keep_is_identity_with_normalization():
    assert preprocess_keep("rumahnya") == "rumahnya"


def test_naive_strip_demonstrates_false_positive_problem():
    assert preprocess_naive_strip("punya tanya hanya Kenya") == "pu ta ha Ke"


def test_sastrawi_clitic_guard_preserves_canonical_false_positives():
    roots = {"rumah", "buku", "pidato"}
    for token in FALSE_POSITIVES:
        assert (
            preprocess_sastrawi_clitic(token, root_dict=roots, remover=SuffixNyaRemover())
            == token
        )


def test_sastrawi_clitic_guard_strips_known_roots():
    roots = {"rumah", "buku", "pidato"}
    text = "rumahnya bukunya pidatonya"
    assert (
        preprocess_sastrawi_clitic(text, root_dict=roots, remover=SuffixNyaRemover())
        == "rumah buku pidato"
    )


def test_sentinel_uses_same_naive_matcher_by_design():
    assert preprocess_sentinel("punya rumahnya") == "pu <NYA> rumah <NYA>"


def test_rule_based_resolver_substitutes_simple_anaphoric_case():
    resolver = RuleBasedNyaResolver(root_dict={"pidato"})
    text = "Sukarno dilahirkan di Surabaya. Pidatonya terkenal."
    assert resolver.resolve(text) == "Sukarno dilahirkan di Surabaya. Pidato Sukarno terkenal."


def test_rule_based_resolver_ignores_dictionary_guard_failures():
    resolver = RuleBasedNyaResolver(root_dict={"buku"})
    assert resolver.resolve("Dia punya buku.") == "Dia punya buku."


# --- Real PySastrawi integration tests (skipped if Sastrawi not installed) ---
# These guard against drift between the Sastrawi adapter and the real package.
# The existing tests above use the SuffixNyaRemover test double and would not catch a
# broken import path or a Protocol-mismatch in create_sastrawi_remover().


def _make_real_sastrawi_remover():
    pytest.importorskip("Sastrawi")
    from nya_ir.preprocessing.sastrawi import create_sastrawi_remover

    return create_sastrawi_remover()


def test_sastrawi_real_adapter_satisfies_filter_protocol():
    """create_sastrawi_remover() must return an object exposing .filter(token) -> str."""
    remover = _make_real_sastrawi_remover()
    assert hasattr(remover, "filter"), "Sastrawi adapter does not expose .filter()"
    result = remover.filter("rumahnya")
    assert isinstance(result, str)


@pytest.mark.parametrize("token", FALSE_POSITIVES)
def test_sastrawi_real_adapter_with_small_dict_preserves_false_positives(token):
    """With a small dictionary, the guard must preserve canonical Indonesian false positives.

    Even if the underlying Sastrawi behaviour over-strips internally, the dictionary guard
    in preprocess_sastrawi_clitic must reject any candidate root that is not in the dict.
    """
    remover = _make_real_sastrawi_remover()
    assert (
        preprocess_sastrawi_clitic(token, root_dict=SMALL_TEST_ROOTS, remover=remover)
        == token
    ), f"Real Sastrawi adapter modified {token!r} despite the dictionary guard"


@pytest.mark.parametrize("token", KNOWN_ROOT_TOKENS)
def test_sastrawi_real_adapter_with_small_dict_strips_known_roots(token):
    """With a small dictionary, true clitic forms whose root is in the dict must be stripped."""
    remover = _make_real_sastrawi_remover()
    expected = token[:-3]  # rumahnya -> rumah, bukunya -> buku, pidatonya -> pidato
    assert (
        preprocess_sastrawi_clitic(token, root_dict=SMALL_TEST_ROOTS, remover=remover)
        == expected
    ), f"Real Sastrawi adapter failed to strip clitic from {token!r}"


def test_sastrawi_real_adapter_handles_mixed_sentence():
    """Mixed sentence with one false positive and one true clitic — both must be handled correctly."""
    remover = _make_real_sastrawi_remover()
    text = "Dia punya rumahnya"
    assert (
        preprocess_sastrawi_clitic(text, root_dict=SMALL_TEST_ROOTS, remover=remover)
        == "Dia punya rumah"
    )


# --- Path A: Sastrawi-bundled root dictionary ---
# Production runs use Sastrawi's full ~30k-word Indonesian root dictionary as root_dict.
# This is the "what real practitioners using Sastrawi would experience" framing for Strategy 3.


def test_load_sastrawi_root_dictionary_returns_nonempty_set():
    pytest.importorskip("Sastrawi")
    from nya_ir.preprocessing.sastrawi import load_sastrawi_root_dictionary

    roots = load_sastrawi_root_dictionary()
    assert isinstance(roots, frozenset)
    # Sastrawi's bundled dict has ~30k entries; assert a conservative lower bound
    assert len(roots) > 10_000


def test_load_sastrawi_root_dictionary_contains_canonical_roots():
    """Confirms Path A behavior: 'biasa' is in the dict, so 'biasanya' will be stripped."""
    pytest.importorskip("Sastrawi")
    from nya_ir.preprocessing.sastrawi import load_sastrawi_root_dictionary

    roots = load_sastrawi_root_dictionary()
    # Roots that should be present
    for word in ["biasa", "punya", "rumah", "buku", "pidato", "karena"]:
        assert word in roots, f"Expected {word!r} in Sastrawi's bundled dict"
    # Non-roots (substrings or made-up words) should be absent
    assert "pu" not in roots
    assert "xyzqq" not in roots


def test_path_a_dictionary_strips_biasanya():
    """Path A specification: with Sastrawi's full dict, biasanya gets stripped to biasa.

    This documents the known limitation of Strategy 3 under Path A — adverbial -nya forms
    like biasanya/karenanya are stripped because their roots are in the dictionary.
    """
    pytest.importorskip("Sastrawi")
    from nya_ir.preprocessing.sastrawi import (
        create_sastrawi_remover,
        load_sastrawi_root_dictionary,
    )

    roots = load_sastrawi_root_dictionary()
    remover = create_sastrawi_remover()
    assert preprocess_sastrawi_clitic("biasanya", root_dict=roots, remover=remover) == "biasa"


def test_path_a_dictionary_preserves_true_false_positives():
    """Path A still preserves words whose stripped form is not a known root (punya, Kenya, etc.)."""
    pytest.importorskip("Sastrawi")
    from nya_ir.preprocessing.sastrawi import (
        create_sastrawi_remover,
        load_sastrawi_root_dictionary,
    )

    roots = load_sastrawi_root_dictionary()
    remover = create_sastrawi_remover()
    # punya → strips to "pu", "pu" not in dict → preserved
    # Kenya → strips to "Ke", "ke" not in dict (after casefold) → preserved
    # Sonya → strips to "So", not in dict → preserved
    for token in ["punya", "tanya", "hanya", "Kenya", "Sonya", "Tanya"]:
        assert (
            preprocess_sastrawi_clitic(token, root_dict=roots, remover=remover) == token
        ), f"Path A unexpectedly modified {token!r}"


# --- Strategy 5 (RuleBasedNyaResolver) edge cases ---
# These tests pin down current resolver behavior so future refactors can't silently change it.
# Where current behavior is debatable, the test docstring flags it as a design question.


def test_resolver_no_antecedent_in_window_leaves_text_unchanged():
    """When no candidate antecedent exists in the prior 2 sentences, -nya stays as-is.

    The classifier falls through to POSSESSIVE (no antecedent → not anaphoric), so no
    substitution happens. This is correct conservative behavior.
    """
    resolver = RuleBasedNyaResolver(root_dict={"pidato"})
    text = "Pidatonya menggemparkan."
    assert resolver.resolve(text) == text


def test_resolver_picks_first_capitalized_word_in_window():
    """Documents current behavior: ``CAPITALIZED_NP_RE.search()`` returns the FIRST match.

    For a sentence ``Sukarno bertemu dengan Hatta``, the resolver picks ``Sukarno`` as the
    antecedent (not ``Hatta``). Whether this is the correct heuristic for Indonesian
    anaphora is debatable — Indonesian discourse often resolves anaphora to the most
    RECENT mention, not the first. Flagged as a design question for the 3-5h Phase 1 sprint.
    """
    resolver = RuleBasedNyaResolver(root_dict={"pidato"})
    text = "Sukarno bertemu dengan Hatta. Pidatonya menggemparkan."
    assert resolver.resolve(text) == "Sukarno bertemu dengan Hatta. Pidato Sukarno menggemparkan."


def test_resolver_window_excludes_sentences_beyond_two():
    """Antecedent window is hard-coded to 2 sentences; earlier sentences are out of scope.

    Strong assertion: the resolver picks ``Banyak`` (sentence-initial common noun in the most
    recent in-window sentence — known limitation), NOT ``Sukarno`` (out of window).
    """
    resolver = RuleBasedNyaResolver(
        root_dict={"pidato"}, antecedent_window_sentences=2
    )
    # Sukarno is 3 sentences back from Pidatonya — out of window.
    text = "Sukarno datang. Cuaca cerah. Banyak orang. Pidatonya menggemparkan."
    result = resolver.resolve(text)
    # Sukarno must not be selected (out of window)
    assert "Pidato Sukarno" not in result
    # Resolver picks first capitalized word in most recent in-window sentence ("Banyak orang")
    assert result == "Sukarno datang. Cuaca cerah. Banyak orang. Pidato Banyak menggemparkan."


def test_sastrawi_adapter_filter_is_idempotent_under_repeated_calls():
    """Defensive: confirm the visitor is stateless under repeated `.filter()` calls.

    The reviewer flagged this as a P1 concern (mutable visitor object), but the underlying
    Sastrawi visitor's ``remove()`` is a pure ``re.sub`` call — no state. This test
    documents that property so a future Sastrawi upgrade with stateful internals would
    immediately fail.
    """
    remover = _make_real_sastrawi_remover()
    # Same input, two calls → same output
    assert remover.filter("rumahnya") == remover.filter("rumahnya")
    # Different inputs interleaved must not contaminate each other
    a1 = remover.filter("rumahnya")
    b = remover.filter("punya")
    a2 = remover.filter("rumahnya")
    assert a1 == a2 == "rumah"
    assert b == "pu"  # the visitor IS dictionary-blind; outer guard in strategies.py is what protects punya


def test_resolver_substitutes_first_nya_correctly_in_multi_nya_passage():
    """The first anaphoric -nya in a passage gets substituted with a real proper-noun antecedent.

    Substitutions happen in reverse offset order so positions don't shift mid-replacement.
    """
    resolver = RuleBasedNyaResolver(root_dict={"pidato", "karya"})
    text = "Sukarno datang. Pidatonya menggemparkan. Karyanya juga terkenal."
    result = resolver.resolve(text)
    # The first -nya correctly references Sukarno (the only proper noun in window)
    assert "Pidato Sukarno" in result


def test_resolver_chained_nya_skips_prior_nya_as_antecedent():
    """The chained-nya sweep: -nya forms must NOT be picked as antecedent candidates.

    Pre-sweep behaviour: ``Karyanya`` resolved to ``Karya Pidatonya`` because the regex
    matched ``Pidatonya`` as a capitalised NP candidate. Post-sweep: the resolver iterates
    candidates and skips any ending in ``-nya``, falling back to the next eligible
    capitalised NP. With ``Sukarno`` reachable in the older sentence, ``Karyanya`` now
    resolves to ``Karya Sukarno``.
    """
    resolver = RuleBasedNyaResolver(root_dict={"pidato", "karya"})
    text = "Sukarno datang. Pidatonya menggemparkan. Karyanya juga terkenal."
    result = resolver.resolve(text)
    # The first -nya still resolves to Sukarno (unchanged behaviour)
    assert "Pidato Sukarno" in result
    # The second -nya now correctly skips "Pidatonya" and falls back to "Sukarno"
    assert "Karya Sukarno" in result
    # And the nonsense chain is gone
    assert "Karya Pidatonya" not in result


def test_resolver_chain_of_three_falls_back_gracefully_when_all_candidates_are_nya():
    """When every in-window candidate is a -nya form, the resolver gives up cleanly.

    Window=2 sentences. For ``Bukunya`` at the end, the prior 2 sentences contain only
    ``Pidatonya`` and ``Karyanya`` as capitalised candidates — both -nya forms, both
    skipped. ``Sukarno`` is out of window. Resolver returns None → POSSESSIVE → no
    substitution. Conservative and honest behaviour.
    """
    resolver = RuleBasedNyaResolver(
        root_dict={"pidato", "karya", "buku"}, antecedent_window_sentences=2
    )
    text = "Sukarno bicara. Pidatonya bagus. Karyanya hebat. Bukunya laris."
    result = resolver.resolve(text)
    # Pidatonya resolves correctly (Sukarno is in its window)
    assert "Pidato Sukarno" in result
    # Karyanya falls back through Pidatonya skip to Sukarno (still in window=2)
    assert "Karya Sukarno" in result
    # Bukunya: window covers only [Pidatonya, Karyanya] sentences; both candidates skipped → no substitution
    assert "Bukunya" in result, "Bukunya should remain unsubstituted when no proper-noun antecedent is in window"


def test_resolver_known_limitation_picks_sentence_initial_common_noun():
    """KNOWN LIMITATION: capitalized sentence-initial common nouns are picked as antecedents.

    The CAPITALIZED_NP_RE heuristic does not distinguish proper nouns from sentence
    starters. ``Cuaca`` (weather) at the start of a sentence is structurally
    indistinguishable from ``Sukarno`` (proper noun). Without POS tagging this is
    irreducible. Documented for the limitations section of the paper.
    """
    resolver = RuleBasedNyaResolver(root_dict={"pidato"})
    text = "Cuaca cerah. Pidatonya menggemparkan."
    result = resolver.resolve(text)
    # The resolver wrongly substitutes "Cuaca" as the antecedent for Pidatonya.
    assert result == "Cuaca cerah. Pidato Cuaca menggemparkan."

