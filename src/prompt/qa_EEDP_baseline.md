### Information:
    #Text_Information: {{info1}}
    #Table_Information: {{info2}}
    Question: {{info3}}

### Instructions
    Please carefully analyze the above provided information, perform necessary numerical calculations, and provide accurate answers to the given question using the provided data. Take into account the pre-text, table, and post-text when formulating your response.
    When performing numerical calculations, ensure you gather the required information and follow a step-by-step approach.
    1. Elicit the most relevant domain fact or knowledge that might be useful for you to extract the right operands and forming the right approach to the problem.  
    2. Translate table data into sentences and identify gold evidence for answering the question.  
    3. Define a sequence of atomic operations (Add, Subtract, Divide, Multiply, Greater, Min, Max, Exp) which take into account only two operands at a time. Divide a complex task into a sequence of atomic operations defined above.  
    4. Finally, conclude as: The final answer is <answer>.If the question requires a specific numerical answer, ensure the answer is a single value calculated to its final result. If the question requires a year as the answer, ensure the answer is given as a numerical year (e.g., 2006).


### Important Constraint:
    **The final answer must be formatted as follows:
    (1)The content of the final answer must **always** be enclosed within <>, such as `<answer>`. 
    (2)Do not include any separators or currency symbols (e.g., "$62,357" should be written as "62357"). If the result is in percentage form, convert it to decimal form (e.g., 12.5%  should be written as 0.125). If there is division with an infinite decimal, **keep the result to 5 decimal places**, such as “The final answer is <18000>” or “The final answer is <0.12987>”.
    (3)Pay attention to the unit required in the answer to the question. For example, for “What was the total stock-based performance unit awards expense in 2018, in millions?”, the answer should be in millions. That is, if the answer to the above question is 2,000,000, you should respond with "The final answer is <2>."In summary,ensure that the answer is in the units required by the question. However, you do not need to include the unit in your response,only provide the number corresponding to the unit.
    
### Demonstration: 
    #Text_Information:"Effective Income Tax Rate”: A reconciliation of the United States federal statutory income tax rate to our effective income tax rate is as follows: In 2019 and 2018 we had pre-tax losses of $19,573 and $25,403, respectively, which are available for carry forward to offset future taxable income."
    
    #Table_Information:"
    <table border="1" style="border-collapse: collapse; width: 50%;">
    <tr>
        <th colspan="2">Year Ended</th>
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
    Response #:  
    Domain Knowledge: Pre-tax losses, or operating losses, refer to financial losses that a company incurs before considering the effects of income taxes. To find the 2019 percentage change in pre-tax losses, we need to find the difference between the new and the old value of the pre-tax losses, then divide the obtained difference by the old value and multiply this value by 100.
    Gold Evidences:  
        • The pre-tax losses in 2019 are $19,573.  
        • The pre-tax losses in 2018 are $25,403.
    Steps:  
            1. Subtract 25403 from 19573.
                • Response 1: 19573-25403=-5830  
            2. Divide #1 by 25403.  
                • Response 2: -5830/25403=-0.22950  
    The final answer is <-0.22950>

### Response #: