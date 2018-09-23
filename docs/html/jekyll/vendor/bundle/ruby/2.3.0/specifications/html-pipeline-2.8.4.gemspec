# -*- encoding: utf-8 -*-
# stub: html-pipeline 2.8.4 ruby lib

Gem::Specification.new do |s|
  s.name = "html-pipeline"
  s.version = "2.8.4"

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib"]
  s.authors = ["Ryan Tomayko", "Jerry Cheung"]
  s.date = "2018-07-24"
  s.description = "GitHub HTML processing filters and utilities"
  s.email = ["ryan@github.com", "jerry@github.com"]
  s.homepage = "https://github.com/jch/html-pipeline"
  s.licenses = ["MIT"]
  s.post_install_message = "-------------------------------------------------\nThank you for installing html-pipeline!\nYou must bundle Filter gem dependencies.\nSee html-pipeline README.md for more details.\nhttps://github.com/jch/html-pipeline#dependencies\n-------------------------------------------------\n"
  s.rubygems_version = "2.5.2.1"
  s.summary = "Helpers for processing content through a chain of filters"

  s.installed_by_version = "2.5.2.1" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<activesupport>, [">= 2"])
      s.add_runtime_dependency(%q<nokogiri>, [">= 1.4"])
    else
      s.add_dependency(%q<activesupport>, [">= 2"])
      s.add_dependency(%q<nokogiri>, [">= 1.4"])
    end
  else
    s.add_dependency(%q<activesupport>, [">= 2"])
    s.add_dependency(%q<nokogiri>, [">= 1.4"])
  end
end
