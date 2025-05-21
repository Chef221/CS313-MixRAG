### Information:
#Text_Information: {{info1}}
#Table_Information: {{info2}}
Question: {{info3}}

### Instructions:
Please carefully analyze the provided information, perform necessary numerical calculations, and provide accurate answers to the given question using the provided data. Follow the steps outlined below to derive the answer from the provided information:

1. **Restate the Question:**
    Clearly restate what the question is asking and identify the nature of the answer required:
    - Numerical value (e.g., 12345)
    - A specific year (e.g., 2006)
    - A decimal number (e.g., 0.12345)
    - A yes/no (true/false) question type

2. **Extract Relevant Data:**
    From the **#Text_Information** and **#Table_Information** provided, extract and list all data points directly relevant to answering the question.

3. **Compute the Answer:**
    Using the relevant data, perform the necessary calculations or logical reasoning step-by-step. Ensure the reasoning is clear and explicit, showing intermediate steps as needed.

4. **Output the Final Result:**
    Only output the following two parts:
    - Final Answer: `The final answer is <answer>`
    - Calculation Formula: `The complete calculation formula is ##[calculation formula]##`

    **Note:**
    - For numerical values, years, and decimal numbers, the answer should only include numbers without any units or symbols. Decimal numbers should be kept to a maximum of five decimal places.
    - For yes/no questions, the answer should be `<yes>` or `<no>`.
    - The calculation formula should only contain mathematical expressions using `*` for multiplication and `/` for division, without any additional text or units.

### Example:
```plaintext
#Text_Information:"Effective Income Tax Rate”: A reconciliation of the United States federal statutory income tax rate to our effective income tax rate is as follows: In 2019 and 2018 we had pre-tax losses of 19573 and 25403, respectively, which are available for carry forward to offset future taxable income."

#Table_Information:"
<table border='1' style='border-collapse: collapse; width: 50%;'>
<tr>
    <th colspan='2'>Year Ended</th>
</tr>
<tr>
    <th>December 31, 2018</th>
    <th>December 31, 2019</th>
</tr>
<tr>
    <td>United States federal statutory rate</td>
    <td>21.00%</td>
    <td>21.00%</td>
</tr>
<tr>
    <td>Effective income tax rate</td>
    <td>1.99%</td>
    <td>-0.01%</td>
</tr>
</table>"

Question: What was the 2019 percentage change in pre-tax losses?
