### Information:
    #Text_Information: {{info1}}
    #Table_Information: {{info2}}
    Question: {{info3}}

### Instructions:
    Please carefully analyze the above provided information. Let’s think step by step to solve the Quesion.
    Finally, conclude as: `The final answer is <answer>`.If the question requires a specific numerical answer, ensure the answer is a single value calculated to its final result. If the question requires a year as the answer, ensure the answer is given as a numerical year (e.g., The final answer is <2006>). Provide the complete calculation formula enclosed in ##, as follows: The complete calculation formula is ##[calculation formula]##. If the answer can be solved without writing out the calculation formula, then just give the final unique numerical result.   

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
    Step 1: Understanding the Question  
    The question asks about the **percentage change in pre-tax losses** from 2018 to 2019. To calculate this, we need to find the difference between the pre-tax losses in 2019 and 2018, then express this difference as a percentage of the 2018 pre-tax losses.

    Step 2: Identify Relevant Data  
    From the **Text_Information**, we learn that pre-tax losses in:
    - **2019**: 19,573  
    - **2018**: 25,403

    Step 3: Calculate the Difference Between Pre-Tax Losses  
    Now, let’s calculate the difference in pre-tax losses between 2019 and 2018:  
    - 2019 pre-tax losses: 19,573  
    - 2018 pre-tax losses: 25,403  

    We subtract 2019's pre-tax losses from 2018's pre-tax losses:  
    **Difference = 19,573 - 25,403 = -5,830**

    Step 4: Calculate the Percentage Change  
    Now that we have the difference, we need to express this as a percentage of the 2018 pre-tax losses. To do this, we divide the difference by the 2018 pre-tax losses and multiply by 100.  
    The formula to calculate percentage change is:  
    \[
    \text{Percentage Change} = \frac{\text{Difference}}{\text{2018 Pre-Tax Losses}} = \frac{-5,830}{25,403}
    \]

    Step 5: Perform the Calculation  
    Now, let's compute the percentage change:
    \[
    \frac{-5,830}{25,403} = -0.22950
    \]

    Step 6: Final Answer  
    The percentage change in pre-tax losses is -0.22950. Since the question asks for the answer in decimal form, we present the result as a negative decimal.

    **The final answer is <-0.22950>
    **The complete calculation formula is ##(19573 - 25403) / 25403##

### Important Constraint:
    **The final answer and complete calculation formula must be formatted as follows:
    (1)The content of the final answer must **always** be enclosed within <>, such as `<answer>`. 
    (2)Do not include any separators or currency symbols (e.g., "$62,357" should be written as "62357"). If the result is in percentage form, convert it to decimal form (e.g., 12.5% should be written as 0.125). If there is division with an infinite decimal, **keep the result to 5 decimal places**, such as "The final answer is <0.12987>".
    (3)Pay attention to the unit required in the answer to the question. For example, for “What was the total stock-based performance unit awards expense in 2018, in millions?”, the answer should be in millions. That is, if the answer to the above question is 2,000,000, you should respond with "The final answer is <2>."In summary,ensure that the answer is in the units required by the question. However, you do not need to include the unit in your response,only provide the number corresponding to the unit.
    (4)Ensure that the calculation formula is enclosed in ##, and what is enclosed is only a calculation formula that can fully derive the final answer, and does not contain any other information.
    (5)For multiplication and division in the calculation formula, use * for multiplication and / for division. The numbers in the calculation formula should not include any units (e.g., $12.5) or separator punctuation (e.g., 20,000).