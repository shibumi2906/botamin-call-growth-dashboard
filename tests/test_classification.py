import unittest

import pandas as pd

from src.classification import classify_calls, _next_step_analysis, _detect_relevance
from src.data_processing import clean_transcript
from src.export import prepare_export_df
from src.recommendations import generate_ab_test
from src.metrics import build_funnel_table, calculate_metrics


class ClassificationTests(unittest.TestCase):
    def classify(self, text):
        return classify_calls(pd.DataFrame([dict(transcript_clean=clean_transcript(text),
            has_transcript=True, duration_seconds=60, termination_reason='')]))

    def test_refusal_does_not_pass_acceptance_in_funnel(self):
        result = self.classify('bot: Давайте назначим демо. user: Не интересно, не звоните.')
        row = result.iloc[0]
        self.assertEqual(row['call_outcome'], 'do_not_call')
        self.assertEqual(row['funnel_stage_reached'], '5_next_step_proposed')
        self.assertFalse(row['next_step_accepted'])
        funnel = build_funnel_table(result).set_index('stage')
        for stage in ['6_next_step_accepted', '7_context_collected']:
            self.assertEqual(funnel.loc[stage, 'reached_at_least'], 0)
            self.assertEqual(funnel.loc[stage, 'step_conversion_pct'], 0)
        metrics = calculate_metrics(result)
        self.assertEqual(metrics['Proposal → Acceptance Conversion'], 0)
        self.assertEqual(metrics['Qualified Next Step Rate'], 0)
        self.assertIn('outcome_evidence', prepare_export_df(result))

    def test_cancellation_after_bot_confirmation_revokes_acceptance(self):
        for refusal in ['Отмените встречу.', 'Я передумал.', 'Не звоните.', 'Не интересно.']:
            with self.subTest(refusal=refusal):
                result = self.classify('bot: Давайте назначим демо. user: Завтра. '
                    'bot: Отмечаю встречу на завтра. user: ' + refusal)
                row = result.iloc[0]
                self.assertFalse(row['next_step_accepted'])
                self.assertEqual(row['acceptance_evidence'], '')
                self.assertEqual(row['acceptance_proposal_evidence'], '')
                self.assertEqual(row['funnel_stage_reached'], '5_next_step_proposed')
                self.assertFalse(row['is_qualified_next_step'])
                self.assertEqual(build_funnel_table(result).iloc[6]['reached_at_least'], 0)

    def test_acceptance_and_outcome_are_separate(self):
        for answer, stage in [('Да.', '6_next_step_accepted'), ('Завтра.', '7_context_collected')]:
            result = self.classify('bot: Давайте назначим демо. user: ' + answer)
            self.assertEqual(result.iloc[0]['call_outcome'], 'accepted')
            self.assertEqual(result.iloc[0]['funnel_stage_reached'], stage)
            self.assertEqual(build_funnel_table(result).iloc[6]['step_conversion_pct'], 100)
        row = self.classify('user: Не звоните.').iloc[0]
        self.assertEqual(row['call_outcome'], 'do_not_call')
        self.assertEqual(row['funnel_stage_reached'], '0_no_dialog')

    def test_new_acceptance_after_refusal(self):
        result = self.classify('bot: Давайте назначим демо. user: Нет. '
                              'bot: Отправить материалы? user: На почту.')
        self.assertTrue(result.iloc[0]['next_step_accepted'])
        self.assertEqual(result.iloc[0]['call_outcome'], 'accepted')
        self.assertIn('материалы', result.iloc[0]['acceptance_proposal_evidence'])

    def test_mixed_funnel_counts_only_valid_acceptance(self):
        result = pd.concat([
            self.classify('bot: Давайте назначим демо. user: Не интересно, не звоните.'),
            self.classify('bot: Давайте назначим демо. user: Завтра.'),
            self.classify('bot: Давайте назначим демо. user: Завтра. '
                          'bot: Спасибо. user: Отмените встречу.'),
        ], ignore_index=True)
        funnel = build_funnel_table(result).set_index('stage')
        self.assertEqual(funnel.loc['5_next_step_proposed', 'reached_at_least'], 3)
        self.assertEqual(funnel.loc['6_next_step_accepted', 'reached_at_least'], 1)
        self.assertEqual(funnel.loc['6_next_step_accepted', 'step_conversion_pct'], 33.33)
        self.assertEqual(funnel.loc['7_context_collected', 'reached_at_least'], 1)
        self.assertEqual(calculate_metrics(result)['Proposal → Acceptance Conversion'], 33.33)

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
