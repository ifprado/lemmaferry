# LemmaFerry: provisional name and GitHub launch

Research checked on 2026-10-16, Pacific time. Domain endpoint checks began at 2026-10-17T03:22:04Z, equivalent to October 16 at 8:22 p.m. Pacific. These checks are observations, not reservations or purchases.

## Recommended name

**LemmaFerry**, repository slug `lemmaferry`, subtitle: **Verified mathematics, measurable transfer.**

The ferry metaphor describes carrying verified mathematical structure into a general model. It suits an experimental toolkit without promising AGI. The initial README description can be:

> LemmaFerry is an open research toolkit for generating Lean-checked synthetic tasks and measuring whether mathematical training transfers to language-model reasoning and internal representations.

| Candidate | Collision search | Domain observation | Decision |
| --- | --- | --- | --- |
| LemmaFerry | Exact-name web search found no relevant existing project. GitHub public repository API returned `total_count: 0`, `incomplete_results: false`. | Official Verisign `.com` RDAP returned HTTP 404 for `lemmaferry.com`. This indicates no matching registry record at the check, subject to endpoint availability and registry restrictions. The `.org` registry endpoint returned 403, so `.org` status is unknown. | Recommended working name. Check registrar checkout before deciding to buy. |
| AxiomFerry | Exact-name web search found no relevant existing project. GitHub public repository API returned `total_count: 0`, `incomplete_results: false`. | Official Verisign `.com` RDAP returned HTTP 404 for `axiomferry.com`. | Backup. |
| LemmaVoyage | Exact-name web search found no relevant existing project. GitHub public repository API returned `total_count: 0`, `incomplete_results: false`. | Official Verisign `.com` RDAP returned HTTP 404 for `lemmavoyage.com`. | Second backup. |
| ProofHarbor | An active company, ProofHarbor Federal Integrations, uses the name. | Not investigated further. | Avoid. |

No exact-name search is a trademark clearance. A missing search result or missing DNS record does not establish domain availability. A positive control against the same Verisign endpoint for `example.com` returned HTTP 200 with `ldhName: EXAMPLE.COM` and `objectClassName: domain`, confirming the registry service was reachable. No domain was purchased, and registrar price or ability to complete registration was not verified. GitHub Pages works without purchasing a domain, so obtaining one does not block the experiment.

Direct records and sources:

- [LemmaFerry `.com` registry endpoint](https://rdap.verisign.com/com/v1/domain/lemmaferry.com)
- [LemmaFerry `.org` registry endpoint](https://rdap.publicinterestregistry.org/rdap/domain/lemmaferry.org)
- [AxiomFerry `.com` registry endpoint](https://rdap.verisign.com/com/v1/domain/axiomferry.com)
- [LemmaVoyage `.com` registry endpoint](https://rdap.verisign.com/com/v1/domain/lemmavoyage.com)
- [GitHub exact candidate repository query](https://api.github.com/search/repositories?q=lemmaferry)
- [Verisign explanation of its registry RDAP service](https://www.verisign.com/news-insights/registration-data-access-protocol/help/)
- [ICANN explanation of RDAP registration information](https://www.icann.org/rdap/)
- [Existing ProofHarbor company](https://proofharborfederal.com/)

## First GitHub upload

Assumption: the supplied starter is a local directory named `lemmaferry` and is not already inside a Git repository. The user creates the remote under their own GitHub account; there is no need to invent an account name or choose an organization yet.

1. Review `README.md`, the roadmap, and the license. Edit in vim if desired: `vim README.md`.
2. On GitHub, create an **empty public repository** named `lemmaferry`. Do not select the README, license, or gitignore initialization options because the starter supplies those files.
3. In a terminal inside the extracted directory, run the commands below. If the starter is already a Git repository, skip `git init` and inspect its existing branch and remotes before proceeding.

```bash
cd lemmaferry
git init -b main
git status --short
git add .
git diff --cached --stat
git commit -m "Start LemmaFerry reproducible research toolkit"
```

4. Copy the remote URL from GitHub. The following commands deliberately use a placeholder; replace `YOUR-USER` with the account owning the new repository before running them.

```bash
git remote add origin https://github.com/YOUR-USER/lemmaferry.git
git remote -v
git push -u origin main
```

5. Invite the cousin through the repository's collaborator settings after confirming their GitHub username. Keep this as a user action; no invitation has been sent.
6. Add repository topics such as `lean4`, `synthetic-data`, `representation-learning`, `theorem-proving`, and `reproducible-research`.
7. Open three initial issues: run the first Lean validation; reproduce a shuffled-anchor control; choose the first language-model pair and exact evaluation task.

GitHub's official instructions cover [creating a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository) and [uploading locally hosted code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github). GitHub CLI users can instead authenticate normally and run `gh repo create lemmaferry --public --source=. --remote=origin --push` after reviewing the local commit.

## Static project report on GitHub Pages

Use this only for the static project explanation and experiment reports. The Python/Lean experiments run locally or in CI, not in GitHub Pages.

1. Put a self-contained `index.html` report in `docs/` and commit it. Put `.nojekyll` in `docs/` if the report should be served directly without Jekyll processing.
2. Open repository **Settings > Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Choose branch **main**, folder **/docs**, then **Save**.
5. Once published, the default project address is `https://YOUR-USER.github.io/lemmaferry/`. A purchased domain can be connected later through Pages settings and DNS.

Source: [GitHub Pages publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site). The current documentation permits Pages for public repositories on GitHub Free. It supports publishing the repository root or `/docs` from a branch.

## Open-source defaults

- Use **Apache-2.0** for original project code, with the complete official license in `LICENSE`. SPDX confirms this exact identifier. This is a practical project default, not a conclusion that every imported asset can use the same license.
- Keep the permissions and attribution for models, source corpora, Lean libraries, and copied code separate. Record origins and original licenses in a small provenance table before redistributing any third-party artifact.
- Include a `CITATION.cff`, a concise `CONTRIBUTING.md`, an issue template for experiment reports, and a clear `reproduce` command. Publish failed experiments and the controls alongside positive results.
- Keep any personal source conversations and the original scanned sketch outside the public starter unless the user decides to publish them.

Primary license sources: [Apache's official license and application instructions](https://www.apache.org/licenses/LICENSE-2.0), [plain text license](https://www.apache.org/licenses/LICENSE-2.0.txt), and [SPDX Apache-2.0](https://spdx.org/licenses/Apache-2.0.html).

The starter also contains `name-check.json` with a second independent registry check and its exact UTC timestamp. No registrar transaction was attempted.
