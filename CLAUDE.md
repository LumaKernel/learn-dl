@AGENTS.md

## Architecture

- Effect.ts をコアライブラリとして使用
- イミュータブルな設計を基本とする
- Effect の tagged class パターン（`Data.TaggedClass`, `Schema.TaggedClass`）でドメインモデルを定義する
- サービスは `Context.Tag` + `Layer` で構成し、依存注入する
- エラーは tagged union (`Data.TaggedError`) で型安全に扱う
- `interface` キーワードは使わない。代わりに `type` を使う

## Workflow

- 各ステップごとに commit, push, CI確認を行う
