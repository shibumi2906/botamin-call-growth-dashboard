import unittest

import pandas as pd

from src.classification import classify_calls, _next_step_analysis, _detect_relevance
from src.data_processing import clean_transcript
from src.export import prepare_export_df
from src.recommendations import generate_ab_test


class ClassificationTests(unittest.TestCase):
    def test_order_roles_and_answer_context(self):
        cases = [
            ('user: Да. bot: Давайте назначим демо.', False),
            ('bot: ' + 'Подробности. ' * 100 + 'Давайте назначим демо. user: Да.', True),
            ('bot: Давайте назначим демо. user: Нет, не подходит.', False),
            ('bot: Давайте назначим демо. user: Когда-нибудь.', False),
            ('bot: Давайте назначим демо. user: Никуда не готов.', False),
            ('bot: Давайте назначим демо. bot: Удобно говорить? user: Да.', False),
            ('bot: Давайте назначим демо. user: Завтра. bot: Отправить материалы? user: Нет.', False),
            ('bot: Давайте назначим демо. user: Завтра. bot: Спасибо, до завтра.', True),
            ('bot: Давайте назначим демо. user: Завтра. bot: Отмечаю встречу на завтра. Какая у вас база? user: Не знаю.', True),
            ('bot: Давайте назначим демо. user: Да. user: Нет, передумал.', False),
            ('bot: Давайте назначим демо. Хорошо, завтра.', False),
            ('Давайте назначим демо. Да.', False),
            ('Клиент: Да.\nБот: Отправить материалы?\nКлиент: На почту.', True),
            ('assistant: Давайте назначим демо; client: Хорошо.', True),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                result = _next_step_analysis(clean_transcript(text))
                self.assertEqual(result['next_step_accepted'], expected)
                if expected:
                    self.assertTrue(result['acceptance_evidence'].startswith('user:'))
                    self.assertTrue(result['acceptance_proposal_evidence'].startswith('bot:'))

    def test_bot_cannot_supply_relevance(self):
        for text in ['bot: У нас есть отдел продаж.', 'user: У нас нет отдела продаж.',
                     'bot: Интересно? user: Не интересно.']:
            self.assertFalse(_detect_relevance(text)[0])
        self.assertTrue(_detect_relevance('bot: Хорошо. user: Я директор.')[0])

    def test_pipeline_and_evidence_export(self):
        text = clean_transcript('bot: Давайте назначим демо.\nuser: Да.')
        self.assertIn('\n', text)
        result = classify_calls(pd.DataFrame([dict(transcript_clean=text, has_transcript=True,
                                                  duration_seconds=60, termination_reason='')]))
        self.assertEqual(result.iloc[0]['funnel_stage_reached'], '6_next_step_accepted')
        self.assertTrue(result.iloc[0]['next_step_accepted'])
        self.assertIn('acceptance_evidence', prepare_export_df(result))

    def test_all_effects_are_hypotheses(self):
        for kind, stage in [('technical', 'before_contact'), ('offer', 'offer_to_relevance_signal'),
                            ('risk', 'offer_to_relevance_signal'), ('next_step', 'proposal_to_acceptance'),
                            ('script', 'contact_to_consent'), ('script', 'consent_to_offer'),
                            ('qualification', 'acceptance_to_context_or_finish')]:
            result = generate_ab_test(dict(recommendation_type=kind, affected_stage=stage), {})
            self.assertIn('Гипотеза:', result['expected_effect'])
            self.assertNotRegex(result['expected_effect'], r'\d')
            if kind == 'risk':
                self.assertEqual(result['main_metric'], 'Qualified Next Step Rate')


if __name__ == '__main__':
    unittest.main()
