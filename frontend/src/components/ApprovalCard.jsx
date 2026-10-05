import DiffView from "./DiffView";

export default function ApprovalCard({ request, loading, onDecide }) {
  if (request?.type === "push_approval") {
    return (
      <section className="card warn">
        <h2>Approve push &amp; pull request</h2>
        <p>{request.message}</p>
        {request.tests === "no_tests" && (
          <p className="error">Warning: no automated tests were found for this repository.</p>
        )}
        <dl className="kv">
          <dt>Branch</dt>
          <dd>{request.branch || "—"}</dd>
          <dt>Commit</dt>
          <dd><code>{request.commit || "—"}</code></dd>
        </dl>
        <Buttons loading={loading} onDecide={onDecide} approve="Push & open PR" reject="Keep local only" />
      </section>
    );
  }

  return (
    <section className="card warn">
      <h2>Approve code change</h2>
      <p>
        <code>{request?.path || "unknown file"}</code>
      </p>
      <DiffView original={request?.original_content} proposed={request?.content} />
      <Buttons loading={loading} onDecide={onDecide} approve="Approve" reject="Reject" />
    </section>
  );
}

function Buttons({ loading, onDecide, approve, reject }) {
  return (
    <div className="actions">
      <button className="primary" disabled={loading} onClick={() => onDecide("approve")}>
        {approve}
      </button>
      <button className="danger" disabled={loading} onClick={() => onDecide("reject")}>
        {reject}
      </button>
    </div>
  );
}
