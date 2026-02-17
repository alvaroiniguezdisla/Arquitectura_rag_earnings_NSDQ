
import re

text = "Net sales were $50.5 billion, up 5%."
text_lower = text.lower()
print(f"Text: '{text_lower}'")

# 1. Simple 'sales'
p0 = r"sales"
m0 = re.search(p0, text_lower)
print(f"1. Match 'sales': {m0}")

# 2. 'sales were'
p1 = r"sales\s+were"
m1 = re.search(p1, text_lower)
print(f"2. Match 'sales were': {m1}")

# 3. 'sales were $'
p2 = r"sales\s+were\s+\$"
m2 = re.search(p2, text_lower)
print(f"3. Match 'sales were $': {m2}")

# 4. 'sales were $50.5'
p3 = r"sales\s+were\s+\$?([\d\.]+)"
m3 = re.search(p3, text_lower)
print(f"4. Match 'sales were $50.5': {m3}")
if m3: print(f"   Group 1: {m3.group(1)}")

# 5. Full Original
p_orig = r"sales\s+(?:of|was|reached|totaled|is)\s+\$?([\d\.]+)\s+billion"
m_orig = re.search(p_orig, text_lower)
print(f"5. Match Original: {m_orig}")
