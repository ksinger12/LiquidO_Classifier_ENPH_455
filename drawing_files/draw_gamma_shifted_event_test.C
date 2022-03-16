{
int numberOfImages = 5000;
std::vector<TCanvas*> canvases(numberOfImages);
for (int i=0; i<numberOfImages; i++) {
  TCanvas *c = new TCanvas(("c" + std::to_string(i)).c_str(), std::to_string(i).c_str(),700,600);
  canvases[i] = c;
  c->SetLogz();
  gStyle->SetOptStat(0);

  TRandom3 ranshi;
  ranshi.SetSeed(0);
  Float_t xs;
  Float_t ys;
  xs = ranshi.Uniform()*40.0-20.0;
  ys = ranshi.Uniform()*40.0-20.0;


  TH2F *liquido_event = new TH2F("liquido_event","liquido_event",200,-100+xs,100+xs,200,-100+ys,100+ys);
  liquido_event->SetTitle("");
  liquido_event->SetTickLength(0.,"X");
  liquido_event->SetTickLength(0.,"Y");
  liquido_event->SetLabelSize(0.,"X");
  liquido_event->SetLabelSize(0.,"Y");
  std:string entry = "Entry$==";
  entry += std::to_string(i);
  op_hits->Draw("h_pos_y/10.:h_pos_x/10.>>liquido_event",entry.c_str(),"col");

  //randx = rand();
  //randy = rand();
  //op_hits->Draw("(h_pos_y+randy)/10.:(h_pos_x+randx)/10.>>liquido_event",entry.c_str(),"col");

  std::string filenameTrain = "/home/kylesinger/simulation/build/images/rotated_images/Test/Gamma_Ray/img_"+std::to_string(i)+".jpg";
  c->SaveAs(filenameTrain.c_str());
  c->Close();
  delete c;
  delete liquido_event;
 }
}
