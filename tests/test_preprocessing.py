from nya_ir.preprocessing import (
    RuleBasedNyaResolver,
    SuffixNyaRemover,
    preprocess_keep,
    preprocess_naive_strip,
    preprocess_sastrawi_clitic,
    preprocess_sentinel,
)


FALSE_POSITIVES = ["punya", "tanya", "hanya", "biasanya", "Kenya", "Sonya", "Tanya"]


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

