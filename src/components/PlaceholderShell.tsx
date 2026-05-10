type PlaceholderShellProps = {
  message: string;
};

function PlaceholderShell({ message }: PlaceholderShellProps) {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <section className="mx-auto flex min-h-screen max-w-5xl flex-col items-center justify-center gap-6 px-6 text-center">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-8 shadow-2xl">
          <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Desktop App Shell</p>
          <h1 className="mt-4 text-4xl font-semibold">Rich Media Viewer</h1>
          <p className="mt-4 text-slate-300">{message}</p>
        </div>
      </section>
    </main>
  );
}

export default PlaceholderShell;
