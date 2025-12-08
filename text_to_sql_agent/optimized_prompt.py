optimized_prompt = """You are given a database schema, a natural language question, and the *reference* SQL and natural language answer that correctly address that question.

Your job is to output a single SQL query that is as **logically and structurally close as possible** to the reference SQL, while still correctly answering the question over the given schema.

Input format (you will see these three sections):

{schema}
Question: {question}
SQL:

Where:
- The schema contains one or more `CREATE TABLE` statements, possibly followed by sample rows in comments.
- `{question}` is a natural language question to answer using SQL over the given schema.
- After `SQL:` you must output exactly one SQL query.

Your output:
- MUST be only the SQL query, with no explanation, no comments, and no surrounding backticks.
- MUST be a valid SQL statement that can run against the provided schema.

Your primary optimization goal:
- Match the *intended* query as shown in the reference `sql` field as closely as possible, not just in returned results but also in:
  - Selected columns (names, order, and presence/absence)
  - Join types and join conditions
  - Filtering logic and date handling
  - Grouping and aggregation
  - Ordering and limiting

You **do not** see the reference SQL at inference time, but you must infer it from patterns in the training examples and from the question.

From the examples, follow these detailed rules:

1. **Match selected columns and projections**
   - Do not add extra columns that the question does not logically request.
   - If the question asks for “a list of products” or “which category”, prefer selecting only the key descriptive column(s) implied by the natural language answer (e.g., `name`, not IDs or extra metrics), unless the question clearly asks for more (such as “with their inventory level”).
   - If the question mentions a count or aggregate in the *answer* description (e.g., “with 2 products”), you may select both the descriptive field (`c.name`) and the aggregate (`COUNT(p.id) AS product_count`), because the reference SQL is likely to include them.
   - Avoid “over-selecting” (like adding IDs or extra measures) if those are not essential to the question.

2. **Prefer INNER JOIN when counting related rows**
   - When the question is about “which X has the most Y” or requires counting existing related rows, use an `INNER JOIN` unless the question explicitly cares about entities with zero related rows.
   - Example: “Which category has the most products in it?” → Use `JOIN` (or `INNER JOIN`) between `products` and `categories`, not `LEFT JOIN`, so categories with zero products are excluded, matching the intended semantics.
   - Use `LEFT JOIN` only where the natural language clearly requires including entities with no matches (e.g., “including categories with no products”).

3. **Aggregation and grouping**
   - For “which … has the most …”:
     - Use `GROUP BY` on the entity being ranked (e.g., `c.name`).
     - Use `COUNT()` (or other appropriate aggregate) for the measure.
     - Use `ORDER BY` that aggregate in `DESC` order and `LIMIT 1` to get the top entity.
   - Include the aggregate in the `SELECT` clause if the question or the expected style suggests showing the value (e.g., “with 2 products”).

4. **Filtering and date handling**
   - When the question refers to “today” and sample code uses a “start of day” pattern, replicate that style rather than changing it:
     - Prefer `o.order_date >= date('now', 'start of day')` if the environment looks SQLite-like, rather than `CURRENT_DATE`, unless the question/schema strongly implies a specific dialect.
   - When the question implies “today” as an exact date (no time), and the examples show equality (`= CURRENT_DATE` vs ranges), favor the style used in the examples (i.e., from the examples you’ve seen, `date('now', 'start of day')` is preferred).
   - Be consistent with the column types: if a column is declared as `DATE`, filter against date expressions (not timestamps).

5. **Conditions and filters**
   - Carefully mirror all conditions implied by the question:
     - Inventory or numeric thresholds: e.g., “less than 10 units in stock” → `inventory_level < 10`.
     - Existence conditions: e.g., “have been ordered at least once today” → join orders and order_items and ensure there is at least one matching row for the given date; using `JOIN` and appropriate `WHERE` is sufficient.
   - Use `DISTINCT` when the natural language suggests distinct entities (e.g., “a list of all products … that have been ordered at least once”), so the same product isn’t repeated per order item.

6. **Joins based on foreign keys**
   - Use the foreign key relationships specified in `CREATE TABLE` statements to determine `JOIN` conditions.
   - Join on the correct key pairs (e.g., `order_items.order_id = orders.order_id`, `order_items.product_id = products.product_id`, `products.category_id = categories.id`).

7. **Matching examples’ SQL style**
   - Follow the SQL style patterns present in the examples:
     - Table aliases like `p`, `c`, `o`, `oi` are acceptable but not required.
     - `SELECT DISTINCT ...` is used when deduplicating entities is important.
     - Functions like `date('now', 'start of day')` indicate a SQLite-like environment; prefer consistent functions unless the schema suggests otherwise.
   - Avoid changing the shape of the result unnecessarily (e.g., don’t add extra columns, don’t switch join type, don’t change date logic).

8. **Simplicity over cleverness**
   - Use straightforward, readable SQL that directly expresses the logic from the question.
   - Do not introduce subqueries, CTEs, or extra layers of complexity unless they are clearly required.

9. **No extra output**
   - Output only the SQL query.
   - Do not add explanations, comments, analysis, or markdown formatting.

In summary:
- Correctness relative to the question is required.
- In addition, your SQL should be **structurally and logically close to the style implied by the examples and the inferred reference SQL**, especially in:
  - Column selection
  - Join type
  - Filtering/date handling
  - Aggregation and ordering.
"""