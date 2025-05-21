### Instructions:
    Given the provided #Text_Information and #Table_Information, directly answer the question in the format:`The final answer is <answer>.`

### Important Constraint:
    **The final answer must be formatted as follows:
    (1)The content of the final answer must **always** be enclosed within <>, such as `<12>`. 
    (2)Do not include any separators or currency symbols (e.g., "$62,357" should be written as "62357"). If the result is in percentage form, convert it to decimal form (e.g., 12.5% should be written as 0.125). If there is division with an infinite decimal, **keep the result to 5 decimal places**, such as "The final answer is <0.12987>".
    (3)Pay attention to the unit required in the answer to the question. For example, for “What was the total stock-based performance unit awards expense in 2018, in millions?”, the answer should be in millions. That is, if the answer to the above question is 2,000,000, you should respond with "The final answer is <2>."In summary,ensure that the answer is in the units required by the question. However, you do not need to include the unit in your response,only provide the number corresponding to the unit.

### Demonstration: 
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

    --------------------------------
    Response #: 
        The final answer is <-0.22950>

### Information:
    #Text_Information: {{info1}}
    #Table_Information: {{info2}}
    Question: {{info3}}

### Response #: