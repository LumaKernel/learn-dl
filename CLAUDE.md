@AGENTS.md

## Architecture

- Effect.ts をコアライブラリとして使用
- イミュータブルな設計を基本とする
- Effect の tagged class パターン（`Data.TaggedClass`, `Schema.TaggedClass`）でドメインモデルを定義する
- サービスは `Context.Tag` + `Layer` で構成し、依存注入する
- エラーは tagged union (`Data.TaggedError`) で型安全に扱う
- `interface` キーワードは使わない。代わりに `type` を使う

## 言語

- コード中のコメント、JSDoc、UIテキスト、コミットメッセージはすべて日本語で書く
- 専門用語には英語を併記する（例: 勾配降下法 (gradient descent)、逆伝播 (backpropagation)）
- 変数名・関数名・型名は英語のまま

## Workflow

- 各ステップごとに commit, push, CI確認を行う
