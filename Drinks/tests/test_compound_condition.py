import re
from django.test import TestCase
from Drinks.complexity_calculator_csharp import calculate_compound_condition_weight

class TestCompoundConditionWeight(TestCase):

    def test_simple_if_condition(self):
        line = "if (x > 10) {"
        self.assertEqual(calculate_compound_condition_weight(line), 1)

    def test_if_with_logical_and(self):
        line = "if (x > 10 && y < 5) {"
        self.assertEqual(calculate_compound_condition_weight(line), 2)  # 1 + 1 (one &&)

    def test_if_with_logical_or(self):
        line = "if (a == b || c != d) {"
        self.assertEqual(calculate_compound_condition_weight(line), 2)  # 1 + 1 (one ||)

    def test_if_with_multiple_conditions(self):
        line = "if (x > 10 && y < 5 || z == 0) {"
        self.assertEqual(calculate_compound_condition_weight(line), 3)  # 1 + 2 (&& and ||)

    def test_while_condition(self):
        line = "while (counter < 100) {"
        self.assertEqual(calculate_compound_condition_weight(line), 1)

    def test_while_with_multiple_conditions(self):
        line = "while (x > 0 && y == 10 || z != 5) {"
        self.assertEqual(calculate_compound_condition_weight(line), 3)  # 1 + 2 (&& and ||)

    def test_do_while_condition(self):
        line = "do { x++; } while (x < 10);"
        self.assertEqual(calculate_compound_condition_weight(line), 1)

    def test_switch_statement(self):
        line = "switch (choice) {"
        self.assertEqual(calculate_compound_condition_weight(line), 1)

    def test_non_condition_statement(self):
        line = "Console.WriteLine(\"Hello, world!\");"
        self.assertEqual(calculate_compound_condition_weight(line), 0)  # Not a condition

    def test_assignment_statement(self):
        line = "x = (a && b) || (c && d);"  # Logical operators but not a condition
        self.assertEqual(calculate_compound_condition_weight(line), 0)  # Should be ignored