from tounum_cls import config


def test_every_class_is_configured_everywhere():
    for mapping in (config.POOL_COUNTS, config.EXT_COUNTS, config.ZIP_FILES):
        assert set(mapping) == set(config.CLASSES)


def test_pool_is_balanced_photo_vs_non_photo():
    positives = config.POOL_COUNTS[config.POSITIVE_CLASS]
    negatives = sum(n for c, n in config.POOL_COUNTS.items() if c != config.POSITIVE_CLASS)
    assert positives == negatives


def test_stages_are_cumulative_and_end_with_all_negative_classes():
    previous = []
    for negatives in config.STAGES.values():
        assert set(previous) < set(negatives) or not previous
        previous = negatives
    assert set(previous) == set(config.CLASSES) - {config.POSITIVE_CLASS}
