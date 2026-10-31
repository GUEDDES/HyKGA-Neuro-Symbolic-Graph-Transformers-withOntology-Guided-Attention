# TSAB Annotation Guidelines, Temporal Split Protocol, and Public Benchmark Transfer Specification

## 1. Ontology Vocabulary and W3C SIOC / FOAF Grounding

The social discourse ontology $\mathcal{O} = (\mathcal{C}, \mathcal{R}, \mathcal{A}, \mathcal{S})$ defines seven pairwise-disjoint atomic entity concepts ($\mathrm{DisjointClasses}(\mathcal{C})$):

| Concept $c \in \mathcal{C}$ | W3C SIOC / Standard Mapping | Span Identification & Annotation Criteria |
|---|---|---|
| `User` | `sioc:UserAccount` / `foaf:Agent` | Explicit user handles (`@username`), author account nodes, and named person accounts participating in threads. |
| `Message` | `sioc:Post` / `sioc:Item` | Root posts, replies, quote posts, and cross-platform submissions/comments (`sioc:Post` on Reddit). |
| `Hashtag` | `sioc:Tag` | Explicit hashtag tokens (`#tag`) and platform flair tags. |
| `Topic` | `sioc:topic` / `skos:Concept` | Multi-word or nominal thematic subjects discussed in the message (e.g., `"renewable energy policy"`). |
| `Location` | `geo:SpatialThing` | Geopolitical entities, cities, countries, venues, or facilities mentioned in text or user profiles. |
| `Event` | `schema:Event` | Named conferences, summits, elections, emergencies, or breaking-news incidents (e.g., `"Geneva Summit"`). |
| `Sentiment` | `marl:Opinion` | Explicit affective or evaluative polarity spans expressed by a message (positive, negative, neutral/mixed). |

---

## 2. Directed Relation Vocabulary and Admissibility Axioms ($\mathcal{A} \cup \mathcal{S}$)

Each positive relation $r \in \mathcal{R}$ is annotated as a directed triple $(e_h, r, e_t)$ subject to open-world TBox domain/range axioms ($\mathcal{A}$) and closed-world instance integrity constraints ($\mathcal{S}$):

1. **`authored_by`** ($\texttt{Message} \times \texttt{User}$): Links a message node $m$ to its creating user account $u$. Functional constraint ($\mathcal{S}$): at most one author per message ($\le 1\ \texttt{authored\_by}.\texttt{User}$).
2. **`mentions`** ($\texttt{Message} \times (\texttt{User} \sqcup \texttt{Hashtag} \sqcup \texttt{Topic} \sqcup \texttt{Event})$): Links a message $m$ to an entity explicitly referenced in its text body.
3. **`located_in`** ($(\texttt{Message} \sqcup \texttt{User} \sqcup \texttt{Event}) \times \texttt{Location}$): Links a geo-tagged message, user profile, or named event to its geographic location.
4. **`expresses_sentiment`** ($\texttt{Message} \times \texttt{Sentiment}$): Links a message $m$ to its primary evaluative polarity span. Functional constraint ($\mathcal{S}$): $\le 1\ \texttt{expresses\_sentiment}.\texttt{Sentiment}$ per message.
5. **`interacts_with`** ($\texttt{User} \times \texttt{User}$): Direct conversational reply or quote interaction from user $u_1$ to user $u_2$.
6. **`follows`** ($\texttt{User} \times \texttt{User}$): Directed subscription link from user $u_1$ to user $u_2$. Irreflexivity constraint ($\mathcal{S}$): $e_h \neq e_t$ ($\mathrm{Irreflexive}(\texttt{follows})$).
7. **`retweets`** ($\texttt{User} \times \texttt{Message}$): Repost/amplification action by user $u$ on message $m$ (mapped to crosspost/boost on Reddit/Mastodon).
8. **`no_relation`**: Assigned to intra-message co-occurring candidate entity pairs $(e_h, e_t)$ that hold no valid relation in $\mathcal{R}$. Stratified during benchmark construction into $60\%$ TBox-admissible type pairs ($\exists r \in \mathcal{R}: \mathbf{m}_{\mathcal{A}}(c_h, c_t)[r] = 1$, requiring semantic disambiguation) and $40\%$ TBox-incompatible type pairs ($\forall r \in \mathcal{R}: \mathbf{m}_{\mathcal{A}}(c_h, c_t)[r] = 0$).

---

## 3. Chronological Thread Partitioning and Anti-Leakage Protocol

1. **Chronological Root-Thread Split (70% / 15% / 15%)**:
   - All conversational threads are ordered by the timestamp $\tau_{root}$ of the thread-initiating root message:
     - **Training split**: Weeks 1–17 ($70\%$)
     - **Validation split**: Weeks 18–20 ($15\%$)
     - **Test split**: Weeks 21–24 ($15\%$)
   - Entire conversation trees are assigned to the split of their root post so that no thread straddles two splits.

2. **Causal Target-Pair-Excised Subgraph Construction (`causal_subgraph_builder.py`, Equation 18)**:
   - When encoding message $t_i$ (timestamp $\tau_i$) for candidate pair $(e_h^{(i)}, e_t^{(i)})$, the admissible edge set is:
     $$E_i^{causal} = \Bigl\{ (u, r, v, t_e) \in E_{hist} \;\Big|\; t_e < \tau_i \;\wedge\; \{u, v\} \neq \{e_h^{(i)}, e_t^{(i)}\} \;\wedge\; (u, r, v, t_e) \notin E_{eval\_targets} \Bigr\}.$$
   - Held-out validation and test relation labels are excluded from the historical edge store $E_{hist}$, and all direct edges between $e_h^{(i)}$ and $e_t^{(i)}$ (any relation, either direction) are excised before $k$-hop retrieval, GAT message passing, centrality snapshot lookup, and Louvain community detection.
