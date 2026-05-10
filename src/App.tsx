import { invoke } from '@tauri-apps/api/core';
import { useEffect, useState } from 'react';
import PlaceholderShell from './components/PlaceholderShell';

function App() {
  const [message, setMessage] = useState('Loading Tauri shell...');

  useEffect(() => {
    invoke<string>('greet', { name: 'Rich Media Viewer' })
      .then(setMessage)
      .catch(() => setMessage('Rich Media Viewer'));
  }, []);

  return <PlaceholderShell message={message} />;
}

export default App;
